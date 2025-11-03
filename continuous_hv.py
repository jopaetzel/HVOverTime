#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:06:23 2025

@author: joni
"""

#%% IMPORTS AND DEFINITIONS

import numpy as np
import hvsrpy

from pathlib import Path
from tqdm import tqdm
from matplotlib.dates import date2num
from obspy import read

# plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

def collect_day_files(sds_root: str | Path) -> tuple[list[list[str]], list[str]]:
    '''
    function to scan sds archive and return data file names and station name
    
    Parameters
    ----------
    sds_root : str
        path to sds archive, first subfolder(s) should be year level

    Returns
    -------
    daily_triplets : list
        list of lists containing three file paths per station and day
    stations : list
        list of station numbers

    '''
    sds_root = Path(sds_root)
    grouped = {}

    for f in sds_root.rglob("*.D/*.D.*"):
        parts = f.name.split('.')
        net, sta, loc, chan, _, year, doy = parts
        key = (net, sta, year, doy)
        grouped.setdefault(key, {})[chan] = f

    component_sets = [{'DPZ', 'DPN', 'DPE'}, {'DHZ', 'DHN', 'DHE'}]
    daily_triplets = []

    for file_dict in grouped.values():
        for comp_set in component_sets:
            if comp_set.issubset(file_dict):
                triplet = [str(file_dict[c]) for c in sorted(comp_set)]
                daily_triplets.append(triplet)

    stations = sorted({k[1] for k in grouped.keys()})
    return daily_triplets, stations

def extract_times(file: str, dt: int, n_windows: int) -> list:
    '''
    reads metadata of mseed file, extracts start and end time and computes 
    timesteps in between
    
    Parameters
    ----------
    file : str
        data file to read
    dt : int
        spacing of time steps
    n_windows: int
        number of time windows
    

    Returns
    -------
    list
        matplotlib times for plotting

    '''
    
    st = read(file, headonly=True)
    tr = st[0]
    start = tr.stats.starttime

    # Generate midpoints: first midpoint + windows-1 more spaced by dt
    times = [start + dt/2 + i*dt for i in range(n_windows)]

    # Convert to matplotlib times
    matplotlib_times = [date2num(t.datetime) for t in times]
    return matplotlib_times

#%% PATHS AND COLLECT FILES

sds = '/media/joni/Elements/gunnuvher/mseed/'         # path to sds archive
results_dir = '/home/joni/Desktop/'                   # path to store results

triplet_lists, stations = collect_day_files(sds)
print(f"Found data for {len(stations)} stations: {stations}")

#%% SET PARAMETERS

preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
preprocessing_settings.detrend = "linear"
preprocessing_settings.window_length_in_seconds = 600
preprocessing_settings.orient_to_degrees_from_north = 0.0
preprocessing_settings.filter_corner_frequencies_in_hz = (0.01, 100)
preprocessing_settings.ignore_dissimilar_time_step_warning = False

processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
processing_settings.window_type_and_width = ("tukey", 0.01)
processing_settings.smoothing=dict(operator="konno_and_ohmachi",
                               bandwidth=40,
                               center_frequencies_in_hz=np.geomspace(1, 100, 500))
processing_settings.method_to_combine_horizontals = "geometric_mean"
processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

#%% PROCESS ALL STATIONS IN LIST

for station in tqdm(stations, desc='Stations Processed'):
    amplitudes, times, peak_amp, f0 = [], [], [], []
    filtered = [day_list for day_list in sorted(triplet_lists) if station in day_list[0]]

    for day_list in tqdm(filtered, desc='Days Processed', leave=False):
        try:
            fnames = [day_list[0], day_list[1], day_list[2]]
            srecords = hvsrpy.read([fnames])
            srecords_preprocessed = hvsrpy.preprocess(srecords, preprocessing_settings)
            hvsr = hvsrpy.process(srecords_preprocessed, processing_settings)
        
            hvsr.update_peaks_bounded(search_range_in_hz=(1, 5))    
        
            peak_amp.append(hvsr.peak_amplitudes)
            f0.append(hvsr.peak_frequencies)
            amplitudes.append(hvsr.amplitude)
            times.append(extract_times(day_list[0], 
                                       dt=preprocessing_settings.window_length_in_seconds,
                                       n_windows=len(srecords_preprocessed)))
        except:
            print(f'Failed with {day_list}\n Skipping ...')
            continue
        
    amplitudes = np.vstack(amplitudes)
    times = np.hstack(times)
    peak_amp = np.hstack(peak_amp)
    f0 = np.hstack(f0)
    
    outpath = f'{results_dir}hv_result_{station}.npz'

    np.savez(
        outpath,
        amplitudes=amplitudes,
        times=times,
        peak_amp=peak_amp,
        f0=f0,
        f=hvsr.frequency
    )
    
    # print(f'\nSaved results for {station} -> {outpath}')

