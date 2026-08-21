# -*- coding: utf-8 -*-
"""
Bayesian posterior using GP surrogate + emcee.
- Trains a GP surrogate: (Temp_Bias, Prcp_Fac) -> mean MB (over ref period)
- Uses emcee to infer posterior of parameters given geodetic MB
- Propagates to posterior MB envelopes (weighted by posterior draws)
- Optional: re-run hydrology for a few posterior draws (heavy)
"""

import os, json, math, glob, traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import emcee

from oggm import cfg, utils, workflow, tasks

# ----------------------------
# User config
# ----------------------------
FLOWLINE_TYPE = "centerline"
BASE_SAVE_DIR = "/home/public/sch005506/OGGM/MonteCarloTest-yrq/output"
RUN_HYDROLOGY = False
N_POST_SAMPLES = 200
N_HYDRO_SAMPLES = 60
RUNOFF_MIN_YS = 1980
RUNOFF_YE = 2020

SIGMA_OBS = 150.0
JITTER = 50.0

# ----------------------------
# Helpers
# ----------------------------
def prepro_base_url(flowline_type="centerline"):
    if flowline_type == "centerline":
        return "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/centerlines/W5E5/"
    elif flowline_type == "elevation_band":
        return "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/"
    else:
        raise ValueError("flowline_type must be centerline or elevation_band")

def find_all_glacier_dirs(root):
    paths = []
    for rid_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(rid_dir): 
            continue
        rid = os.path.basename(rid_dir)
        summary = os.path.join(rid_dir, f"{rid}_mc_mb_summary.csv")
        if os.path.isfile(summary):
            paths.append((rid, rid_dir))
    return sorted(paths)

def build_gp(X, y):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 0.1], length_scale_bounds=(1e-3, 1e3)) \
             + WhiteKernel(noise_level=max(1.0, JITTER**2), noise_level_bounds=(1e-3, 1e6))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)
    gp.fit(X, y)
    return gp

def gpr_predict_mean_std(gp, X):
    m, s = gp.predict(X, return_std=True)
    s = np.maximum(s, 1.0)
    return m, s

def log_prior(params, smp):
    tb, pf = params
    tb_mu, tb_sd = float(np.mean(smp["Temp_Bias"])), float(np.std(smp["Temp_Bias"]) + 1e-6)
    pf_mu, pf_sd = float(np.mean(smp["Prcp_Fac"])), float(np.std(smp["Prcp_Fac"]) + 1e-6)
    
    # Normal priors
    logp_tb = -0.5 * ((tb - tb_mu) / (3 * max(tb_sd, 1.0))) ** 2
    logp_pf = -0.5 * ((pf - pf_mu) / (3 * max(pf_sd, 0.1))) ** 2
    
    return logp_tb + logp_pf

def log_likelihood(params, gp, mb_obs):
    tb, pf = params
    X_test = np.array([[tb, pf]])
    mu_pred, sd_pred = gpr_predict_mean_std(gp, X_test)
    sigma_tot = np.sqrt(sd_pred[0]**2 + SIGMA_OBS**2)
    return -0.5 * ((mb_obs - mu_pred[0]) / sigma_tot) ** 2 - np.log(sigma_tot)

def log_probability(params, gp, mb_obs, smp):
    lp = log_prior(params, smp)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, gp, mb_obs)

# ----------------------------
# OGGM init
# ----------------------------
cfg.initialize(logging_level="WARNING")
cfg.PARAMS["use_multiprocessing"] = True
cfg.PARAMS["continue_on_error"] = True
cfg.PARAMS["store_model_geometry"] = True
work_dir = utils.gettempdir(dirname="/home/public/sch005506/OGGM/GP-emcee", reset=True)
cfg.PATHS["working_dir"] = work_dir
print(f"OGGM working_dir: {work_dir}")

purl = prepro_base_url(FLOWLINE_TYPE)

pairs = find_all_glacier_dirs(BASE_SAVE_DIR)
print(f"Found {len(pairs)} glacier result folders")

overview = []
for rid, rid_dir in pairs:
    try:
        print(f"Processing RID: {rid}")

        smp_path = os.path.join(rid_dir, f"{rid}_mc_mb_summary.csv")
        smp = pd.read_csv(smp_path)
        if smp.empty:
            print("  SKIP: empty summary")
            continue

        # training data
        X = smp[["Temp_Bias", "Prcp_Fac"]].values
        y = smp["Mean_MB"].values

        # read MB time series
        ts_path = os.path.join(rid_dir, f"{rid}_mc_mb_timeseries.csv")
        mb_ts_df = pd.read_csv(ts_path).set_index("Year") if os.path.isfile(ts_path) else None
        years = mb_ts_df.index.values if mb_ts_df is not None else None

        # init gdir to read geodetic obs
        gdirs = workflow.init_glacier_directories([rid],
            from_prepro_level=3, prepro_base_url=purl, prepro_border=80, reset=False, force=False)
        gdir = gdirs[0]
        if not gdir.has_file("mb_calib"):
            try:
                tasks.mb_calibration_from_geodetic_mb(gdir, calibrate_param2='temp_bias')
            except Exception as e:
                print(f"  WARN: mb_calib missing and calibration failed: {e}")

        mb_obs = None
        if gdir.has_file("mb_calib"):
            mbcal = gdir.read_json("mb_calib")
            mb_obs = mbcal.get("reference_mb", mbcal.get("mb_geodetic", None))
        if mb_obs is None:
            print("  SKIP: No geodetic MB")
            continue

        # build GP surrogate
        gp = build_gp(X, y)
        print("  GP kernel:", gp.kernel_)

        # emcee sampling
        nwalkers = 16
        ndim = 2
        
        # Initial positions based on MC samples
        pos = np.column_stack([
            np.random.normal(np.mean(smp["Temp_Bias"]), np.std(smp["Temp_Bias"]), nwalkers),
            np.random.normal(np.mean(smp["Prcp_Fac"]), np.std(smp["Prcp_Fac"]), nwalkers)
        ])
        
        burnin_steps = 300
        prod_steps = 700
        
        # Run MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(gp, mb_obs, smp)
        )
        
        # Burn-in
        print("  Running burn-in...")
        state = sampler.run_mcmc(pos, burnin_steps, progress=True)
        sampler.reset()
        
        # Production run
        print("  Running production...")
        sampler.run_mcmc(state, prod_steps, progress=True)
        
        # Extract samples
        samples = sampler.get_chain(flat=True)
        print(f"  Posterior draws: {samples.shape[0]}")
        
        # Save posterior samples
        tb_draws = samples[:, 0]
        pf_draws = samples[:, 1]
        df_post = pd.DataFrame({"Temp_Bias": tb_draws, "Prcp_Fac": pf_draws})
        df_post.to_csv(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_samples.csv"), index=False)

        # Weighted MB time series using posterior samples
        if mb_ts_df is not None:
            params = smp[["Temp_Bias", "Prcp_Fac"]].values
            ts_mat = mb_ts_df.values.T
            qw = []
            tb_scale = np.std(params[:, 0]) + 1e-6
            pf_scale = np.std(params[:, 1]) + 1e-6
            
            # Use a subset of posterior samples
            n_use = min(N_POST_SAMPLES, len(tb_draws))
            indices = np.random.choice(len(tb_draws), n_use, replace=False)
            
            for idx in indices:
                tbi, pfi = tb_draws[idx], pf_draws[idx]
                ztb = (params[:, 0] - tbi) / tb_scale
                zpf = (params[:, 1] - pfi) / pf_scale
                w = np.exp(-0.5 * (ztb * ztb + zpf * zpf))
                w /= (w.sum() + 1e-12)
                qw.append(w @ ts_mat)
            
            qw = np.stack(qw, axis=0)
            q05 = np.quantile(qw, 0.05, axis=0)
            q50 = np.quantile(qw, 0.50, axis=0)
            q95 = np.quantile(qw, 0.95, axis=0)
            
            out = pd.DataFrame({"q05": q05, "q50": q50, "q95": q95, "mean": qw.mean(axis=0)}, index=years)
            out.index.name = "Year"
            out.to_csv(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_mb_quantiles.csv"))
            
            # plot
            plt.figure(figsize=(10, 5))
            plt.plot(years, q50, label="median")
            plt.fill_between(years, q05, q95, alpha=0.3, label="5-95%")
            plt.xlabel("Year")
            plt.ylabel("Specific MB (mm w.e.)")
            plt.title(f"{rid} posterior MB (GP+emcee)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_mb_envelope.png"), dpi=180)
            plt.close()

        # Optional: rerun hydrology with posterior samples
        if RUN_HYDROLOGY:
            if not gdir.has_file("model_flowlines"):
                try: 
                    tasks.init_present_time_glacier(gdir)
                except Exception as e: 
                    print("  WARN: init_present_time_glacier:", e)
            q_monthly = None
            rows = []
            
            # Use a subset of posterior samples for hydrology
            n_hydro = min(N_HYDRO_SAMPLES, len(tb_draws))
            hydro_indices = np.random.choice(len(tb_draws), n_hydro, replace=False)
            
            for j, idx in enumerate(hydro_indices):
                tb, pf = float(tb_draws[idx]), float(pf_draws[idx])
                fsuf = f"_gpbayes_{j:03d}"
                gdir.write_pickle({"temp_bias": tb, "prcp_fac": pf}, "mb_calib")
                try:
                    tasks.run_with_hydro(
                        gdir, run_task=tasks.run_from_climate_data,
                        store_monthly_hydro=True, output_filesuffix=fsuf,
                        min_ys=RUNOFF_MIN_YS, ye=RUNOFF_YE, fixed_geometry_spinup_yr=None)
                    if gdir.has_file("model_diagnostics", filesuffix=fsuf):
                        with xr.open_dataset(gdir.get_filepath("model_diagnostics", filesuffix=fsuf)) as ds:
                            ds = ds.load()
                        vlist = ["melt_off_glacier", "melt_on_glacier", "liq_prcp_off_glacier", "liq_prcp_on_glacier"]
                        df = pd.DataFrame({v: (ds[v].to_series() * 1e-9 if v in ds else np.nan) for v in vlist})
                        df["total_runoff"] = df.sum(axis=1)
                        df.to_csv(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_runoff_{j:03d}.csv"))
                        ser = df[["total_runoff"]].rename(columns={"total_runoff": f"s{j:03d}"})
                        if q_monthly is None: 
                            q_monthly = ser.copy()
                        else: 
                            q_monthly = q_monthly.join(ser, how="outer")
                        ann = df["total_runoff"].resample("A").sum(min_count=1)
                        rows.append({"RGI_ID": rid, "Sample": j, "Annual_Runoff_Mt_mean": ann.mean(skipna=True),
                                     "Annual_Runoff_Mt_std": ann.std(skipna=True), "Temp_Bias": tb, "Prcp_Fac": pf})
                except Exception as e:
                    print("  WARN: hydro:", e)
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_runoff_summary.csv"), index=False)
            if q_monthly is not None:
                rq = pd.DataFrame({
                    "q05": q_monthly.quantile(0.05, axis=1),
                    "q50": q_monthly.quantile(0.50, axis=1),
                    "q95": q_monthly.quantile(0.95, axis=1),
                    "mean": q_monthly.mean(axis=1),
                    "std": q_monthly.std(axis=1)})
                rq.to_csv(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_runoff_monthly_quantiles.csv"))
                plt.figure(figsize=(11, 5))
                plt.plot(rq.index.values, rq["q50"].values, label="median")
                plt.fill_between(rq.index.values, rq["q05"].values, rq["q95"].values, alpha=0.3, label="5-95%")
                plt.xlabel("Date")
                plt.ylabel("Total runoff (Mt)")
                plt.title(f"{rid} posterior runoff (GP+emcee)")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(rid_dir, f"{rid}_posterior_gp_emcee_runoff_envelope.png"), dpi=180)
                plt.close()

        overview.append({"RGI_ID": rid, "N_post": int(len(tb_draws))})

    except Exception as e:
        print(f"ERROR: {rid}: {e}")
        traceback.print_exc()

if overview:
    pd.DataFrame(overview).to_csv(os.path.join(BASE_SAVE_DIR, "posterior_gp_emcee_overview.csv"), index=False)
print("DONE: GP+emcee Posterior")