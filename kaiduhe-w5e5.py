# -*- coding: utf-8 -*- 
from oggm import cfg, utils, workflow, tasks, graphics
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import traceback
import xarray as xr
from oggm.core import massbalance, flowline
import seaborn as sns

cache_dir = cfg.CACHE_DIR
print(f"Current OGGM cache directory: {cache_dir}")
# Initialize configuration
cfg.initialize(logging_level='WARNING')
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['continue_on_error'] = True  

# Set working directory
cfg.PATHS['working_dir'] = utils.gettempdir(dirname='/home/public/sch005506/OGGM/W5E5-ts', reset=True)
print(f"Working directory: {cfg.PATHS['working_dir']}")

# Read RGI IDs
with open('/home/public/sch005506/OGGM/rgi_ids1.txt', 'r') as f:
    rgi_ids = [line.strip() for line in f if line.strip()]

# Set output path
save_path = '/home/public/sch005506/OGGM/ts/'
os.makedirs(save_path, exist_ok=True)

# Choose flowline type ('elevation_band' or 'centerline')
flowline_type_to_use = 'centerline'  

load_from_prepro_base_url = True

# Instruction for beginning with existing OGGM's preprocessed directories
if load_from_prepro_base_url:
    # to start from level 3 you can do
    if flowline_type_to_use == 'elevation_band':
        prepro_base_url_L3 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
    elif flowline_type_to_use == 'centerline':
        prepro_base_url_L3 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/centerlines/W5E5/'
    else:
        raise ValueError(f"Unknown flowline type '{flowline_type_to_use}'! Select 'elevation_band' or 'centerline'!")

    gdirs = workflow.init_glacier_directories(rgi_ids,
                                              from_prepro_level=3,
                                              prepro_base_url=prepro_base_url_L3,
                                              prepro_border=80,  # could be 80 or 160
                                              reset=True,
                                              force=True,
                                             )

# 1. Glacier geometry calculations
workflow.execute_entity_task(tasks.glacier_masks, gdirs)
workflow.execute_entity_task(tasks.compute_centerlines, gdirs)
workflow.execute_entity_task(tasks.initialize_flowlines, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
workflow.execute_entity_task(tasks.catchment_area, gdirs)
workflow.execute_entity_task(tasks.catchment_width_geom, gdirs)
workflow.execute_entity_task(tasks.catchment_width_correction, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)

# 2. Mass balance calibration
cfg.PARAMS['prcp_fac'] = 2.0
cfg.PARAMS['use_winter_prcp_fac'] = False
cfg.PARAMS['use_temp_bias_from_file'] = False

calibrated_gdirs = []
mb_calibration_results = []  

for gdir in gdirs:
    try:
        try:
            
            tasks.mb_calibration_from_geodetic_mb(gdir, calibrate_param2='temp_bias')
            calibrated_gdirs.append(gdir)
            print(f"Successfully calibrated {gdir.rgi_id} with temp_bias")
            
            mb_calib_data = gdir.read_json('mb_calib')
            print(mb_calib_data)
            
            mb_calibration_results.append({
                'RGI_ID': gdir.rgi_id,
                'Calibration_Method': 'geodetic_temp_bias',
                'MB_Geodetic': mb_calib_data.get('reference_mb', None),
                'Temp_Bias': mb_calib_data.get('temp_bias', None),
                'Prcp_Fac': mb_calib_data.get('prcp_fac', None),
                'melt_f': mb_calib_data.get('melt_f', None),
                'reference_period':mb_calib_data.get('reference_period', None),
                'Calibration_Success': True
            })
            
        except Exception as e:
            print(f"MB calibration with temp_bias failed for {gdir.rgi_id}: {str(e)}")
            try:
                
                tasks.mb_calibration_from_geodetic_mb(gdir, calibrate_param2='prcp_fac')
                calibrated_gdirs.append(gdir)
                print(f"Successfully calibrated {gdir.rgi_id} with prcp_fac")
                
                mb_calib_data = gdir.read_json('mb_calib')
                
                mb_calibration_results.append({
                    'RGI_ID': gdir.rgi_id,
                    'Calibration_Method': 'geodetic_prcp_fac',
                    'MB_Geodetic': mb_calib_data.get('mb_geodetic', None),
                    'MB_Model': mb_calib_data.get('mb_model', None),
                    'Temp_Bias': mb_calib_data.get('bias', None),
                    'Prcp_Fac': mb_calib_data.get('prcp_fac', None),
                    'Calibration_Success': True
                })
                
            except Exception as e2:
                print(f"MB calibration with prcp_fac failed for {gdir.rgi_id}: {str(e2)}")
                try:
                    
                    tasks.mb_calibration_from_geodetic_mb(gdir)
                    calibrated_gdirs.append(gdir)
                    print(f"Successfully calibrated {gdir.rgi_id} with default settings")
                    
                    mb_calib_data = gdir.read_json('mb_calib')
                    
                    mb_calibration_results.append({
                        'RGI_ID': gdir.rgi_id,
                        'Calibration_Method': 'geodetic_default',
                        'MB_Geodetic': mb_calib_data.get('mb_geodetic', None),
                        'MB_Model': mb_calib_data.get('mb_model', None),
                        'Temp_Bias': mb_calib_data.get('bias', None),
                        'Prcp_Fac': mb_calib_data.get('prcp_fac', None),
                        'Calibration_Success': True
                    })
                except Exception as e3:
                    print(f"Default MB calibration failed for {gdir.rgi_id}: {str(e3)}")
                    try:
                        tasks.mb_calibration_from_scalar_mb(gdir)
                        calibrated_gdirs.append(gdir)
                        print(f"Used scalar MB calibration for {gdir.rgi_id}")
                        
                        mb_calibration_results.append({
                            'RGI_ID': gdir.rgi_id,
                            'Calibration_Method': 'scalar',
                            'MB_Geodetic': None,
                            'MB_Model': None,
                            'Temp_Bias': None,
                            'Prcp_Fac': None,
                            'Calibration_Success': True
                        })
                    except Exception as e4:
                        print(f"All MB calibration failed for {gdir.rgi_id}: {str(e4)}")
                        try:
                            gdir.write_pickle({'prcp_fac': 2.5, 'temp_bias': 1}, 'mb_calib')
                            calibrated_gdirs.append(gdir)
                            print(f"Manually set parameters for {gdir.rgi_id}")
                            
                            mb_calibration_results.append({
                                'RGI_ID': gdir.rgi_id,
                                'Calibration_Method': 'manual',
                                'MB_Geodetic': None,
                                'MB_Model': None,
                                'Temp_Bias': 1,
                                'Prcp_Fac': 2.5,
                                'Calibration_Success': False
                            })
                        except:
                            print(f"Failed to set manual parameters for {gdir.rgi_id}")
                            mb_calibration_results.append({
                                'RGI_ID': gdir.rgi_id,
                                'Calibration_Method': 'failed',
                                'MB_Geodetic': None,
                                'MB_Model': None,
                                'Temp_Bias': None,
                                'Prcp_Fac': None,
                                'Calibration_Success': False
                            })
    except Exception as e:
        print(f"Unexpected error during MB calibration for {gdir.rgi_id}: {str(e)}")
        traceback.print_exc()
       
        mb_calibration_results.append({
            'RGI_ID': gdir.rgi_id,
            'Calibration_Method': 'error',
            'MB_Geodetic': None,
            'MB_Model': None,
            'Temp_Bias': None,
            'Prcp_Fac': None,
            'Calibration_Success': False
        })

if mb_calibration_results:
    mb_df = pd.DataFrame(mb_calibration_results)
    mb_csv_path = os.path.join(save_path, 'mb_calibration_results.csv')
    mb_df.to_csv(mb_csv_path, index=False)
    print(f"Saved mass balance calibration results for {len(mb_calibration_results)} glaciers to {mb_csv_path}")

# 4. Static model initialization
if flowline_type_to_use == 'elevation_band':
    cfg.PARAMS['evolution_model'] = 'SemiImplicit'
elif flowline_type_to_use == 'centerline':
    cfg.PARAMS['evolution_model'] = 'FluxBased'

if calibrated_gdirs:
    y0 = calibrated_gdirs[0].get_climate_info()['baseline_yr_0']
    ye = calibrated_gdirs[0].get_climate_info()['baseline_yr_1'] + 1
else:
    y0 = 1961
    ye = 2020

for gdir in calibrated_gdirs:
    try:
        print(f"Running static initialization for {gdir.rgi_id}")
        tasks.run_from_climate_data(
            gdir,
            min_ys=y0,
            ye=ye,
            fixed_geometry_spinup_yr=None,
            use_inversion_flowlines=True,
            output_filesuffix='_historical'
        )
        print(f"Successfully completed static initialization for {gdir.rgi_id}")
    except Exception as e:
        print(f"Static initialization failed for {gdir.rgi_id}: {str(e)}")
        traceback.print_exc()

# 5. Glacier inversion modeling
if calibrated_gdirs:
    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, calibrated_gdirs)
    
    try:
        workflow.calibrate_inversion_from_consensus(
            calibrated_gdirs,
            apply_fs_on_mismatch=True,
            error_on_mismatch=False,
            filter_inversion_output=True
        )
        print(f"Performed inversion for {len(calibrated_gdirs)} glaciers")
    except Exception as e:
        print(f"Inversion calibration failed: {str(e)}")
        traceback.print_exc()
        
    inversion_tasks = [
        tasks.prepare_for_inversion,
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output
    ]
    
    for task in inversion_tasks:
        try:
            workflow.execute_entity_task(task, calibrated_gdirs)
        except Exception as e:
            print(f"Error during inversion task {task.__name__}: {str(e)}")
            traceback.print_exc()
else:
    print("No calibrated glaciers for inversion modeling")
            
cfg.PARAMS['store_model_geometry'] = True  

# 6. Visualization and data storage
centerline_records = []

for gdir in gdirs:
    try:
        print(f"Processing visualization for {gdir.rgi_id}")
        
        # Plot glacier domain
        graphics.plot_domain(gdir, figsize=(8, 7))
        plt.savefig(os.path.join(save_path, f'domain-{gdir.rgi_id}.png'), dpi=300)
        plt.close()
        print(f"Saved domain plot for {gdir.rgi_id}")

        # Plot centerlines with length labels
        fls = None
        if gdir.has_file('model_flowlines'):
            fls = gdir.read_pickle('model_flowlines')
        elif gdir.has_file('inversion_flowlines'):
            fls = gdir.read_pickle('inversion_flowlines')
        elif gdir.has_file('centerlines'):
            fls = gdir.read_pickle('centerlines')
        
        if fls:
            fig, ax = plt.subplots(figsize=(8, 7))
            graphics.plot_centerlines(gdir, ax=ax, use_flowlines=True, add_downstream=False)
            for fl in fls:
                x, y = fl.line.xy
                length_km = fl.dis_on_line[-1] / 1000  
                mid_idx = len(x) // 2
                ax.text(x[mid_idx], y[mid_idx], f'{length_km:.2f} km', fontsize=8, color='blue')
            plt.savefig(os.path.join(save_path, f'centerlines-{gdir.rgi_id}.png'), dpi=300)
            plt.close()
            print(f"Saved centerlines plot for {gdir.rgi_id}")

            # Store centerline lengths
            flowline_lengths = [fl.dis_on_line[-1] / 1000 for fl in fls]
            record = {'RGI_ID': gdir.rgi_id, 'Total_Length_km': sum(flowline_lengths)}
            for i, length in enumerate(flowline_lengths):
                record[f'Flowline_{i+1}_Length_km'] = length
            centerline_records.append(record)
        else:
            print(f"No flowlines found for {gdir.rgi_id}")
        
        try:
            graphics.plot_catchment_areas(gdir, figsize=(8, 7))
            plt.savefig(os.path.join(save_path, f'area-{gdir.rgi_id}.png'), dpi=300)
            plt.close()
            print(f"Saved catchment areas plot for {gdir.rgi_id}")
        except Exception as e:
            print(f"Error plotting catchment areas for {gdir.rgi_id}: {str(e)}")
        
        try:
            graphics.plot_catchment_width(gdir, corrected=True, figsize=(8, 7))
            plt.savefig(os.path.join(save_path, f'width-{gdir.rgi_id}.png'), dpi=300)
            plt.close()
            print(f"Saved catchment width plot for {gdir.rgi_id}")
        except Exception as e:
            print(f"Error plotting catchment width for {gdir.rgi_id}: {str(e)}")
        
        try:
            graphics.plot_inversion(gdir, figsize=(8, 7))  
            plt.savefig(os.path.join(save_path, f'inversion-{gdir.rgi_id}.png'), dpi=300)
            plt.close()
            print(f"Saved inversion plot for {gdir.rgi_id}")
        except Exception as e:
            print(f"Error plotting inversion for {gdir.rgi_id}: {str(e)}")
        
        print(f"Successfully processed {gdir.rgi_id}")

    except Exception as e:
        print(f"Error processing visualization for {gdir.rgi_id}: {str(e)}")
        traceback.print_exc()
        continue
# Save results to CSV
if centerline_records:
    df = pd.DataFrame(centerline_records)
    csv_path = os.path.join(save_path, 'centerline_lengths.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved centerline lengths for {len(centerline_records)} glaciers to {csv_path}")
else:
    print("No centerline data to save")


print("\nStarting mass balance time series calculation...")


all_mb_ts = []

for gdir in calibrated_gdirs:
    try:
        print(f"\nProcessing mass balance time series for {gdir.rgi_id}")
        
        
        tasks.init_present_time_glacier(gdir)
        
        
        mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
        
        
        climate_info = gdir.get_climate_info()
        y0 = climate_info['baseline_yr_0']
        y1 = climate_info['baseline_yr_1']
        years = np.arange(y0, y1 + 1)
        
        
        fls = gdir.read_pickle('model_flowlines')
        mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
        
        
        glacier_mb = {
            'RGI_ID': gdir.rgi_id,
            'Years': years,
            'MB_ts': mb_ts
        }
        all_mb_ts.append(glacier_mb)
        
           
        plt.figure(figsize=(10, 5))
        plt.plot(years, mb_ts, label='Specific MB', color='blue')
        
        
        window_size = 5
        mb_ma = pd.Series(mb_ts).rolling(window=window_size, center=True).mean()
        plt.plot(years, mb_ma, label=f'{window_size}-year moving avg', color='red', linewidth=2)
        

        plt.title(f'Mass Balance Time Series for {gdir.rgi_id}\n'
                 f'Temp Bias: {glacier_mb.get("temp_bias", "N/A")}, '
                 f'Prcp Fac: {glacier_mb.get("prcp_fac", "N/A")}')
        plt.ylabel('Specific MB (mm w.e.)')
        plt.xlabel('Year')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        
        mb_plot_path = os.path.join(save_path, f'mb_timeseries-{gdir.rgi_id}.png')
        plt.savefig(mb_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mass balance time series plot to {mb_plot_path}")
        
    except Exception as e:
        print(f"Error processing mass balance time series for {gdir.rgi_id}: {str(e)}")
        traceback.print_exc()
        continue


if all_mb_ts:
    
    mb_ts_data = []
    
    for glacier in all_mb_ts:
        for year, mb in zip(glacier['Years'], glacier['MB_ts']):
            mb_ts_data.append({
                'RGI_ID': glacier['RGI_ID'],
                'Year': year,
                'Specific_MB': mb
            })
    
    mb_ts_df = pd.DataFrame(mb_ts_data)
    
    
    mb_ts_csv_path = os.path.join(save_path, 'all_mass_balance_timeseries.csv')
    mb_ts_df.to_csv(mb_ts_csv_path, index=False)
    print(f"\nSaved all mass balance time series data to {mb_ts_csv_path}")
    
    
    for glacier in all_mb_ts:
        single_df = pd.DataFrame({
            'Year': glacier['Years'],
            'Specific_MB': glacier['MB_ts']
        })
        single_csv_path = os.path.join(save_path, f'mb_ts_{glacier["RGI_ID"]}.csv')
        single_df.to_csv(single_csv_path, index=False)
        print(f"Saved individual MB time series for {glacier['RGI_ID']} to {single_csv_path}")
else:
    print("No mass balance time series data to save")
    
    
# Inversion & flowline creation
workflow.execute_entity_task(tasks.prepare_for_inversion, calibrated_gdirs)
workflow.execute_entity_task(tasks.mass_conservation_inversion, calibrated_gdirs)
workflow.execute_entity_task(tasks.filter_inversion_output, calibrated_gdirs)

# Critical: this generates model_flowlines
workflow.execute_entity_task(tasks.init_present_time_glacier, calibrated_gdirs)

# 7.Run runoff model (make sure model_flowlines exists now)
for gdir in calibrated_gdirs:
    if not gdir.has_file('model_flowlines'):
        print(f"Missing model_flowlines for {gdir.rgi_id}, skipping...")
        continue

    try:
        tasks.run_with_hydro(
            gdir,
            run_task=tasks.run_from_climate_data,
            store_monthly_hydro=True,
            output_filesuffix='_runoff',
            min_ys=1961,
            ye=2020,
            fixed_geometry_spinup_yr=1980, 
            
        )
        print(f"{gdir.rgi_id} runoff finish")
    except Exception as e:
        print(f"{gdir.rgi_id} runoff error: {str(e)}")
        traceback.print_exc()
        

for gdir in calibrated_gdirs:
    rgi_id = gdir.rgi_id
    file_id = '_runoff'

    if not gdir.has_file('model_diagnostics', filesuffix=file_id):
        print(f"Missing diagnostics file for {rgi_id}, skipping CSV export.")
        continue

    try:
        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
            ds = ds.load()

        runoff_vars = ['melt_off_glacier', 'melt_on_glacier',
                       'liq_prcp_off_glacier', 'liq_prcp_on_glacier']

        runoff_data = {}
        for var in runoff_vars:
            if var in ds:
                runoff_data[var] = ds[var].to_series() * 1e-9  # Convert from kg to Mt
            else:
                runoff_data[var] = pd.Series(np.nan, index=ds.time.to_index())

        df = pd.DataFrame(runoff_data)
        df.index.name = 'date'
        df['total_runoff'] = df.sum(axis=1)
        
        # Add additional columns (retrieve or calculate these values)
        # Assuming these values can be retrieved from the glacier directory:
        # volume_m3: Volume in cubic meters
        # area_m2: Area in square meters
        # length_m: Length in meters
        # off_area: Off-glacier area (calculated from glacier model)
        # on_area: On-glacier area (calculated from glacier model)
        # Fetch additional variables from the model_diagnostics dataset
        additional_vars = ['volume_m3', 'area_m2', 'length_m', 'off_area', 'on_area']

        for var in additional_vars:
            if var in ds:
                df[var] = ds[var].to_series()  # Add the variable directly to the dataframe
            else:
                df[var] = np.nan  # If variable is not present, set as NaN

        # Save to CSV
        csv_path = os.path.join(save_path, f'{rgi_id}_runoff_mt_year_w5e5.csv')
        df.to_csv(csv_path, float_format='%.6f')
        print(f"{rgi_id} runoff data saved to: {csv_path}")

    except Exception as e:
        print(f"{rgi_id} export error: {str(e)}")

print("\nMonthly runoff data (in Mt) exported for all glaciers.")



print("\nAll processing completed")



