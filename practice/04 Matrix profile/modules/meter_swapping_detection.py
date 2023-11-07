import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions, cutoff, house_idx):
    """
    Split time series into two parts: Head and Tail.

    Parameters
    ---------
    consumptions : dict
        Set of time series.

    cutoff : pandas.Timestamp
        Cut-off point.

    house_idx : list
        Indices of houses.

    Returns
    --------
    heads : dict
        Heads of time series.

    tails : dict
        Tails of time series.
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads, tails, house_idx, m):
    """
    Find the swapped time series pair.

    Parameters
    ---------
    heads : dict
        Heads of time series.

    tails : dict
        Tails of time series.

    house_idx : list
        Indices of houses.

    m : int
        Subsequence length.

    Returns
    --------
    min_score : dict
       Time series pair with minimum swap-score.
    """
    eps = 0.001

    min_score = {}

    min_score = None
    eps = 0.001

    for i in house_idx:
        H_i = heads[f'H_{i}'].iloc[:, 0].astype(float)
        T_i = tails[f'T_{i}'].iloc[:, 0].astype(float)
        
        # Calculate matrix profile for T_i
        mp_i = compute_mp(T_i, m=m, ts2=H_i, exclusion_zone=None)
        mp_i_min = np.nanmin(mp_i['mp']) + eps

        for j in house_idx:
            T_j = tails[f'T_{j}'].iloc[:, 0].astype(float)

            # Calculate matrix profile for T_j
            mp_j = compute_mp(T_j, m=m, ts2=H_i)
            mp_j_min = np.nanmin(mp_j['mp'])
            mp_j['mp'][mp_j['mp'] == np.inf] = np.nan

            score = mp_j_min / mp_i_min

            # Code to record best score
            if min_score is None:
                min_score = {'i': i, 'j': j, 'score': score, 'mp_j': mp_j}
            elif score < min_score['score']:
                min_score = {'i': i, 'j': j, 'score': score, 'mp_j': mp_j}

    return min_score


def plot_consumptions_ts(consumptions, cutoff, house_idx):
    """
    Plot a set of input time series and cutoff vertical line.

    Parameters
    ---------
    consumptions : dict
        Set of time series.

    cutoff : pandas.Timestamp
        Cut-off point.

    house_idx : list
        Indices of houses.
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")