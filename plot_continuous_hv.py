#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:15:20 2025

@author: joni
"""

#%% LOAD RESULT FOR ONE STATION

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.gridspec as gridspec
from cmcrameri import cm

sta = '12107'

with np.load(f'/home/joni/Desktop/hv_result_{sta}.npz') as data:
    amplitudes, times, peak_amp, f0, f = (data[k] for k in ['amplitudes', 'times', 'peak_amp', 'f0', 'f'])
    
#%%

# compute average and std of curves
avg_curve = amplitudes.mean(axis=0)
std_curve = amplitudes.std(axis=0)

# set up figure with 1x3 grid
fig = plt.figure(figsize=(14, 7))
outer_gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 0.1], wspace=0.05)

# left panel top+bottom
left_gs = outer_gs[0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

ax  = fig.add_subplot(left_gs[0, 0])  # top: main panel 
ax3 = fig.add_subplot(left_gs[1, 0], sharex=ax)  # bottom f0 vs time panel

# right panel
right_gs = outer_gs[1].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
ax2 = fig.add_subplot(right_gs[0, 0])
ax2.set_ylim(ax.get_ylim())  # match y axis
fig.add_subplot(right_gs[1, 0]).axis("off")  # hide the bottom right panel

vmax = amplitudes.max()

# main plot
T, F = np.meshgrid(times, f)
pcm = ax.pcolormesh(T, F, amplitudes.T, shading='auto',
                    cmap=cm.batlow, vmax=vmax)

# far right: colorbar
cbar_ax = fig.add_subplot(outer_gs[2])
cbar = fig.colorbar(pcm, cax=cbar_ax, label='HVSR Amplitude')

# scatter of f0 vs time
sc = ax3.scatter(times, f0, c=peak_amp, cmap=cm.batlow, s=8, vmax=vmax)

# format time axis
ax.xaxis.set_visible(False)  # hide x labels on top panel
ax3.xaxis_date()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Hh'))

# Llabels
ax.set_ylabel('Frequency (Hz)')
ax3.set_ylabel('f₀ (Hz)')
ax3.set_xlabel('Time')

# side panel: average spectrum with ±1 std shading
ax2.plot(avg_curve, f, color=cm.batlow.get_under(), label='Mean')
ax2.plot(avg_curve - std_curve, f, color=cm.batlow.get_under(), linestyle=':')
ax2.plot(avg_curve + std_curve, f, color=cm.batlow.get_under(), linestyle=':')

ax2.fill_betweenx(f, avg_curve - std_curve, avg_curve + std_curve,
                  color=cm.batlow.get_over(), alpha=0.8, label="±1 STD"
                  )
       
ax2.set_xlabel('HVSR Amplitude')
ax2.set_ylim(f[0], f[-1])   # ensure full frequency range is visible
ax2.set_yticks([])          # hide y ticks
ax2.set_xlim(0, max(avg_curve + std_curve)*1.1)
ax2.legend(loc='upper right')

# overlay scatter of f0 vs time on main panel
# sc = ax.scatter(
#     times,        
#     f0,        
#     c='lightgrey',   
#     s=0.6
# )

ax3.set_facecolor('lightgrey')
ax2.set_facecolor('lightgrey')

# plot single events in time
events = [
    ('2024-07-30T21:11:09', 'M3.0'),
    ('2024-07-29T06:35:47', 'M3.2'),
    ('2024-07-30T23:13:54', 'M2.8'),
        ]

for time_str, label in events:
    # convert string to matplotlib date number
    t_dt = datetime.fromisoformat(time_str)
    t_num = mdates.date2num(t_dt)
    
    # add vertical lines
    ax.axvline(t_num, color='red', linestyle='--', linewidth=1)
    ax3.axvline(t_num, color='red', linestyle='--', linewidth=1)
    
    # add annotations
    ax.text(t_num, ax.get_ylim()[1]*0.95, label, color='red', rotation=90,
            verticalalignment='top', horizontalalignment='right')
    ax3.text(t_num, ax3.get_ylim()[1]*0.95, label, color='red', rotation=90,
             verticalalignment='top', horizontalalignment='right')


plt.suptitle(f'H/V for Station {sta}')
plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.12, wspace=0.3, hspace=0.3)
plt.show()