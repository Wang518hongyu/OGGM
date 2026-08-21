#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# =======================
# OGGM cache & environment
# =======================
os.environ['OGGM_DOWNLOAD_CACHE'] = '/home/public/sch005506/my_oggm_cache'

# =======================
# Libraries
# =======================
import geopandas as gpd
import oggm.cfg as cfg
from oggm import utils, workflow, tasks, graphics

# =======================
# Initialize OGGM
# =======================
cfg.initialize(logging_level='WARNING')

# RGI region (11 = Central Europe)
rgi_region = '11'

# Working directory
WORKING_DIR = utils.gettempdir('OGGM_Inversion')
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True

# =======================
# Read RGI data
# =======================
path = utils.get_rgi_region_file(rgi_region)
rgidf = gpd.read_file(path)

# Select glaciers (O2Region == 2, Pyrenees)
rgidf = rgidf.loc[rgidf['O2Region'] == '2']

# Sort by area for better multiprocessing efficiency
rgidf = rgidf.sort_values('Area', ascending=False)

# =======================
# Initialize glacier directories (preprocessed data)
# =======================
cfg.PARAMS['border'] = 80
base_url = (
    'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
    'L3-L5_files/2023.3/centerlines/W5E5/'
)

gdirs = workflow.init_glacier_directories(
    rgidf,
    from_prepro_level=3,
    prepro_base_url=base_url
)

# =======================
# Inversion parameters
# =======================
# Glen A (Cuffey & Patterson, 2010)
glen_a = 2.4e-24

# Sliding parameter (Oerlemans, 1997)
fs = 5.7e-20

# =======================
# Sensitivity experiment for Glen A
# =======================
with utils.DisableLogger():

    factors = (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] +
        [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0] +
        [6.0, 7.0, 8.0, 9.0, 10.0]
    )

    for f in factors:
        # ---- Without sliding ----
        suf = '_{:03d}_without_fs'.format(int(f * 10))
        workflow.execute_entity_task(
            tasks.mass_conservation_inversion,
            gdirs,
            glen_a=glen_a * f,
            fs=0
        )
        utils.compile_glacier_statistics(
            gdirs,
            filesuffix=suf,
            inversion_only=True
        )

        # ---- With sliding ----
        suf = '_{:03d}_with_fs'.format(int(f * 10))
        workflow.execute_entity_task(
            tasks.mass_conservation_inversion,
            gdirs,
            glen_a=glen_a * f,
            fs=fs
        )
        utils.compile_glacier_statistics(
            gdirs,
            filesuffix=suf,
            inversion_only=True
        )

print('OGGM working dir:', WORKING_DIR)

# =======================
# Post-processing & plots
# =======================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Example statistics file (factor = 1.0, without sliding)
df = pd.read_csv(
    os.path.join(WORKING_DIR, 'glacier_statistics_010_without_fs.csv'),
    index_col=0
)

# Area-volume scatter plot
ax = df.plot(kind='scatter', x='rgi_area_km2', y='inv_volume_km3')
ax.semilogx()
ax.semilogy()

xlim = [1e-2, 0.7]
ylim = [1e-5, 0.05]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Fit in log-log space
dfl = np.log(df[['inv_volume_km3', 'rgi_area_km2']])
slope, intercept, r_value, p_value, std_err = stats.linregress(
    dfl.rgi_area_km2.values,
    dfl.inv_volume_km3.values
)

print('power:', round(slope, 3))
print('scale:', round(np.exp(intercept), 3))

ax = df.plot(
    kind='scatter',
    x='rgi_area_km2',
    y='inv_volume_km3',
    label='OGGM glaciers'
)
ax.plot(xlim, np.exp(intercept) * (np.array(xlim) ** slope), label='Fitted line')
ax.semilogx()
ax.semilogy()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.legend()

# =======================
# Total volume sensitivity
# =======================
dftot = pd.DataFrame(index=factors)

for f in factors:
    suf = '_{:03d}_without_fs'.format(int(f * 10))
    fpath = os.path.join(WORKING_DIR, 'glacier_statistics{}.csv'.format(suf))
    _df = pd.read_csv(fpath, index_col=0, low_memory=False)
    dftot.loc[f, 'without_sliding'] = _df.inv_volume_km3.sum()

    suf = '_{:03d}_with_fs'.format(int(f * 10))
    fpath = os.path.join(WORKING_DIR, 'glacier_statistics{}.csv'.format(suf))
    _df = pd.read_csv(fpath, index_col=0, low_memory=False)
    dftot.loc[f, 'with_sliding'] = _df.inv_volume_km3.sum()

dftot.plot()
plt.xlabel('Factor of Glen A (default = 1)')
plt.ylabel('Regional volume (km^3)')

# =======================
# Calibration against consensus
# =======================
cdf = workflow.calibrate_inversion_from_consensus(
    gdirs[1:],
    filter_inversion_output=False
)
print(cdf.sum())
print(cdf.iloc[:3])

# =======================
# Distributed ice thickness
# =======================
workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

import xarray as xr
import rioxarray as rioxr

ds = xr.open_dataset(gdirs[0].get_filepath('gridded_data'))
ds.distributed_thickness.plot()

# Export distributed thickness to GeoTIFF
workflow.execute_entity_task(
    tasks.gridded_data_var_to_geotiff,
    gdirs,
    varname='distributed_thickness'
)

# Check GeoTIFF existence
for gdir in gdirs:
    tif_path = os.path.join(gdir.dir, 'distributed_thickness.tif')
    assert os.path.exists(tif_path)

# Plot one GeoTIFF
rioxr.open_rasterio(tif_path).plot()

# =======================
# OGGM built-in plotting examples
# =======================
# Select a few glaciers by RGI ID
rgi_ids = ['RGI60-11.0{}'.format(i) for i in range(3205, 3211)]
sel_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in rgi_ids]

# Plot centerlines and inversion result
graphics.plot_centerlines(sel_gdirs)
graphics.plot_inversion(sel_gdirs[0])

plt.show()
