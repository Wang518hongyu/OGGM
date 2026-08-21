# -*- coding: utf-8 -*-
# Monte Carlo uncertainty analysis for OGGM (single or multiple RGI IDs loaded from CSV)
# Server-ready: non-interactive backend, no Unicode comments, clear I/O, optional runoff MC.

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt

from oggm import cfg, utils, workflow, tasks
from oggm.core import massbalance
import xarray as xr

# ----------------------------
# User configuration
# ----------------------------
INPUT_CSV = "/home/public/sch005506/OGGM/rgi_yrq.txt"  # your CSV with RGI IDs
BASE_SAVE_DIR = "/home/public/sch005506/OGGM/MonteCarloTest-yrq/output"  # output root
FLOWLINE_TYPE = "centerline"  # "centerline" or "elevation_band"

# Monte Carlo settings
NUM_SAMPLES = 100
TEMP_BIAS_MEAN = 0.0    # degC
TEMP_BIAS_STD = 2.0     # degC
PRCP_FAC_MEAN = 2.0     # unitless
PRCP_FAC_STD = 0.1      # unitless

# Toggle runoff MC (heavy). If True, we re-run climate with each sample's mb_calib.
RUN_RUNOFF_MC = False

# Run years for runoff hydrology if enabled (must be within climate baseline)
RUNOFF_MIN_YS = 1980
RUNOFF_YE = 2020

# ----------------------------
# Helper functions
# ----------------------------
def load_rgi_ids_from_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "rgi_id" in cols:
        colname = df.columns[cols.index("rgi_id")]
        ids = df[colname].astype(str).str.strip()
    elif "rgiid" in cols:
        colname = df.columns[cols.index("rgiid")]
        ids = df[colname].astype(str).str.strip()
    else:
        # fallback: first column
        ids = df.iloc[:, 0].astype(str).str.strip()
    ids = [i for i in ids if i]
    if not ids:
        raise ValueError("No RGI IDs found in CSV.")
    return ids

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def pick_flowlines_for_mb(gdir):
    # Prefer model_flowlines; fallback to inversion_flowlines; else centerlines.
    if gdir.has_file("model_flowlines"):
        return gdir.read_pickle("model_flowlines")
    if gdir.has_file("inversion_flowlines"):
        return gdir.read_pickle("inversion_flowlines")
    if gdir.has_file("centerlines"):
        return gdir.read_pickle("centerlines")
    return None

def prepro_base_url(flowline_type="centerline"):
    if flowline_type == "centerline":
        return "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/centerlines/W5E5/"
    elif flowline_type == "elevation_band":
        return "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/"
    else:
        raise ValueError("flowline_type must be 'centerline' or 'elevation_band'.")

# NEW: safe init that skips problematic glaciers
def safe_init_glacier_directories(rgi_ids, **kwargs):
    """Init glacier directories one-by-one, skip IDs that fail (e.g. broken tar files)."""
    ok_gdirs = []
    bad_ids = []

    for rid in rgi_ids:
        try:
            g = workflow.init_glacier_directories([rid], **kwargs)
            # init_glacier_directories returns a list
            if isinstance(g, list):
                ok_gdirs.extend(g)
            else:
                ok_gdirs.append(g)
            print(f"[OK] init gdir for {rid}")
        except Exception as e:
            print(f"[SKIP] {rid} because of error: {repr(e)}")
            traceback.print_exc()
            bad_ids.append(rid)

    print(f"\nInit glacier directories done. Success: {len(ok_gdirs)}, Skipped: {len(bad_ids)}")

    # Log skipped glaciers to a file
    if bad_ids:
        ensure_dirs(BASE_SAVE_DIR)
        skip_log = os.path.join(BASE_SAVE_DIR, "skipped_glaciers.txt")
        with open(skip_log, "w") as f:
            for bid in bad_ids:
                f.write(bid + "\n")
        print(f"Skipped RGI IDs written to: {skip_log}")

    return ok_gdirs, bad_ids

# ----------------------------
# OGGM initialize
# ----------------------------
cfg.initialize(logging_level="WARNING")
# You can set this to False if you want simpler debugging during init
cfg.PARAMS["use_multiprocessing"] = True  # OGGM multiprocessing
cfg.PARAMS["continue_on_error"] = True
cfg.PARAMS["store_model_geometry"] = True

work_dir = utils.gettempdir(dirname="/home/public/sch005506/OGGM/MonteCarloTest", reset=True)
cfg.PATHS["working_dir"] = work_dir
print(f"Working directory: {work_dir}")

# ----------------------------
# IO setup
# ----------------------------
rgi_ids = load_rgi_ids_from_csv(INPUT_CSV)
print(f"Loaded {len(rgi_ids)} RGI IDs from CSV.")

save_root = BASE_SAVE_DIR
ensure_dirs(save_root)

# ----------------------------
# Init glacier directories from preprocessed level 3 (safe version)
# ----------------------------
purl = prepro_base_url(FLOWLINE_TYPE)

# Important: reset=False here, working_dir already reset above
gdirs, bad_ids = safe_init_glacier_directories(
    rgi_ids,
    from_prepro_level=3,
    prepro_base_url=purl,
    prepro_border=80,
    reset=False,
    force=True,
)
print(f"Initialized {len(gdirs)} glacier directories. Skipped {len(bad_ids)} glaciers.")

# ----------------------------
# Prepare minimal assets for MB and (optional) hydrology
# For MB-only MC with MultipleFlowlineMassBalance we don't need inversion every time,
# but we may need flowlines generated at least once.
# ----------------------------
# Try to ensure model_flowlines exist, otherwise init present time glacier (inversion not strictly required for MB).
for gdir in gdirs:
    try:
        fls = pick_flowlines_for_mb(gdir)
        if fls is None or not fls:
            # Fallback: try minimal chain to create inversion_flowlines/model_flowlines
            tasks.prepare_for_inversion(gdir)
            tasks.mass_conservation_inversion(gdir)
            tasks.filter_inversion_output(gdir)
            tasks.init_present_time_glacier(gdir)
    except Exception as e:
        print(f"[WARN] Flowline prep failed for {gdir.rgi_id}: {e}")
        traceback.print_exc()

# ----------------------------
# Monte Carlo per glacier
# ----------------------------
all_mc_rows = []  # summary rows: one per (glacier, sample)
for gdir in gdirs:
    rid = gdir.rgi_id
    out_dir = os.path.join(save_root, rid)
    ensure_dirs(out_dir)

    try:
        climate_info = gdir.get_climate_info()
        y0, y1 = climate_info["baseline_yr_0"], climate_info["baseline_yr_1"]
        years = np.arange(y0, y1 + 1)

        fls = pick_flowlines_for_mb(gdir)
        if fls is None or not fls:
            print(f"[SKIP] No flowlines for {rid}.")
            continue

        print(f"\n[MC] {rid}: years {y0}-{y1}, samples {NUM_SAMPLES}")

        # MB Monte Carlo
        per_sample_mb = []  # store per-sample mean/std to aggregate
        per_sample_mb_ts = []  # store per-sample time series (optional aggregation)
        for i in range(NUM_SAMPLES):
            temp_bias = np.random.normal(TEMP_BIAS_MEAN, TEMP_BIAS_STD)
            prcp_fac = np.random.normal(PRCP_FAC_MEAN, PRCP_FAC_STD)

            mbmod = massbalance.MultipleFlowlineMassBalance(
                gdir,
                bias=temp_bias,
                prcp_fac=prcp_fac
            )
            mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

            row = {
                "RGI_ID": rid,
                "Sample": i + 1,
                "Temp_Bias": temp_bias,
                "Prcp_Fac": prcp_fac,
                "Mean_MB": float(np.mean(mb_ts)),
                "MB_Std": float(np.std(mb_ts))
            }
            all_mc_rows.append(row)
            per_sample_mb.append(row)
            per_sample_mb_ts.append(pd.Series(mb_ts, index=years))

            if (i + 1) % max(1, NUM_SAMPLES // 10) == 0:
                print(f"  -> sample {i+1}/{NUM_SAMPLES}")

        # Save per-sample MB summary
        mb_summary_df = pd.DataFrame(per_sample_mb)
        mb_summary_path = os.path.join(out_dir, f"{rid}_mc_mb_summary.csv")
        mb_summary_df.to_csv(mb_summary_path, index=False)

        # Save per-sample MB time series (wide format: columns=sample)
        mb_ts_df = pd.concat(per_sample_mb_ts, axis=1)
        mb_ts_df.columns = [f"Sample_{i+1:03d}" for i in range(NUM_SAMPLES)]
        mb_ts_df.index.name = "Year"
        mb_ts_path = os.path.join(out_dir, f"{rid}_mc_mb_timeseries.csv")
        mb_ts_df.to_csv(mb_ts_path)

        # Yearly quantiles across samples
        qdf = pd.DataFrame({
            "q05": mb_ts_df.quantile(0.05, axis=1),
            "q50": mb_ts_df.quantile(0.50, axis=1),
            "q95": mb_ts_df.quantile(0.95, axis=1),
            "mean": mb_ts_df.mean(axis=1),
            "std": mb_ts_df.std(axis=1),
        })
        qdf.index.name = "Year"
        mb_q_path = os.path.join(out_dir, f"{rid}_mc_mb_quantiles.csv")
        qdf.to_csv(mb_q_path)

        # Plots for MB uncertainty
        plt.figure(figsize=(9, 5))
        plt.hist(mb_summary_df["Mean_MB"], bins=30)
        plt.xlabel("Mean Specific Mass Balance (mm w.e.)")
        plt.ylabel("Frequency")
        plt.title(f"MB Mean Distribution ({rid})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{rid}_mb_mean_distribution.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(qdf.index.values, qdf["q50"].values, label="median")
        plt.fill_between(qdf.index.values, qdf["q05"].values, qdf["q95"].values, alpha=0.3, label="5-95% band")
        plt.xlabel("Year")
        plt.ylabel("Specific MB (mm w.e.)")
        plt.title(f"MB Time Series Uncertainty ({rid})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{rid}_mb_uncertainty_envelope.png"), dpi=300)
        plt.close()

        # Optional: runoff MC (heavy)
        if RUN_RUNOFF_MC:
            # Ensure model_flowlines present
            if not gdir.has_file("model_flowlines"):
                tasks.init_present_time_glacier(gdir)

            runoff_rows = []   # per-sample annual summaries
            runoff_q_monthly = None  # collect monthly quantiles

            for i in range(NUM_SAMPLES):
                temp_bias = float(mb_summary_df.loc[i, "Temp_Bias"])
                prcp_fac = float(mb_summary_df.loc[i, "Prcp_Fac"])

                # write temp parameters into mb_calib used by run_from_climate_data
                gdir.write_pickle({"temp_bias": temp_bias, "prcp_fac": prcp_fac}, "mb_calib")

                fsuf = f"_mc_runoff_{i+1:03d}"
                try:
                    tasks.run_with_hydro(
                        gdir,
                        run_task=tasks.run_from_climate_data,
                        store_monthly_hydro=True,
                        output_filesuffix=fsuf,
                        min_ys=max(RUNOFF_MIN_YS, y0),
                        ye=min(RUNOFF_YE, y1),
                        fixed_geometry_spinup_yr=None,
                    )
                except Exception as e:
                    print(f"[WARN] Runoff task failed for {rid} sample {i+1}: {e}")
                    traceback.print_exc()
                    continue

                if not gdir.has_file("model_diagnostics", filesuffix=fsuf):
                    print(f"[WARN] Missing diagnostics for {rid} sample {i+1}")
                    continue

                with xr.open_dataset(gdir.get_filepath("model_diagnostics", filesuffix=fsuf)) as ds:
                    ds = ds.load()
                # monthly runoff components (kg/s per time step -> OGGM stores mass; convert to Mt)
                vars_needed = [
                    "melt_off_glacier", "melt_on_glacier",
                    "liq_prcp_off_glacier", "liq_prcp_on_glacier"
                ]
                series = {}
                for v in vars_needed:
                    if v in ds:
                        series[v] = ds[v].to_series() * 1e-9  # kg -> Mt
                    else:
                        series[v] = pd.Series(np.nan, index=ds.time.to_index())
                dfm = pd.DataFrame(series)
                dfm.index.name = "date"
                dfm["total_runoff"] = dfm.sum(axis=1)

                dfm.to_csv(os.path.join(out_dir, f"{rid}_runoff_monthly_sample_{i+1:03d}.csv"), float_format="%.6f")

                # annual summary
                ann = dfm["total_runoff"].resample("A").sum(min_count=1)  # Mt/yr
                runoff_rows.append({
                    "RGI_ID": rid,
                    "Sample": i + 1,
                    "Annual_Runoff_Mt_mean": ann.mean(skipna=True),
                    "Annual_Runoff_Mt_std": ann.std(skipna=True)
                })

                # accumulate monthly quantiles
                dfm = dfm[["total_runoff"]].rename(columns={"total_runoff": f"s{i+1:03d}"})
                if runoff_q_monthly is None:
                    runoff_q_monthly = dfm.copy()
                else:
                    runoff_q_monthly = runoff_q_monthly.join(dfm, how="outer")

            if runoff_rows:
                runoff_summary_df = pd.DataFrame(runoff_rows)
                runoff_summary_df.to_csv(os.path.join(out_dir, f"{rid}_mc_runoff_summary.csv"), index=False)

            if runoff_q_monthly is not None:
                rq = pd.DataFrame({
                    "q05": runoff_q_monthly.quantile(0.05, axis=1),
                    "q50": runoff_q_monthly.quantile(0.50, axis=1),
                    "q95": runoff_q_monthly.quantile(0.95, axis=1),
                    "mean": runoff_q_monthly.mean(axis=1),
                    "std": runoff_q_monthly.std(axis=1),
                })
                rq.to_csv(os.path.join(out_dir, f"{rid}_mc_runoff_monthly_quantiles.csv"))

                # plot monthly envelope
                plt.figure(figsize=(11, 5))
                plt.plot(rq.index.values, rq["q50"].values, label="median")
                plt.fill_between(rq.index.values, rq["q05"].values, rq["q95"].values, alpha=0.3, label="5-95% band")
                plt.xlabel("Date")
                plt.ylabel("Total runoff (Mt)")
                plt.title(f"Runoff Monthly Uncertainty ({rid})")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{rid}_runoff_monthly_uncertainty.png"), dpi=300)
                plt.close()

        print(f"[DONE] {rid}")

    except Exception as e:
        print(f"[ERROR] {rid}: {e}")
        traceback.print_exc()
        continue

# Save global MC summary (all glaciers x samples)
if all_mc_rows:
    all_df = pd.DataFrame(all_mc_rows)
    all_df.to_csv(os.path.join(save_root, "all_glaciers_mc_mb_summary.csv"), index=False)
    print(f"\nSaved: {os.path.join(save_root, 'all_glaciers_mc_mb_summary.csv')}")

print("\nAll processing completed.")
