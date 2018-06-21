# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 09:33:24 2017

This script contains plotting methods for gesture analysis

@author: colleen.s
"""

import numpy as np
import math
import colorlover as cl
import plotly.graph_objs as go
from plotly import tools
from plotly.graph_objs import Marker
import random
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE 
from sklearn.metrics import roc_curve, auc

import utility_methods as um
import model_analysis_methods as mam

def get_scatter_plot_trace_color_grp(data):
    """Creats a scatter plot trace given a data frame containing plotting data

    Args:
        data: A data frame containin the following columns
            x - Data for the x axis
            y - Data for the y axis
            grp - group indicators
            clr - string rbc values used to color the plot
    Returns:
        A trace to plot with plotly

    """

    return go.Scatter(
    x = data['x'],
    y = data['y'],
    mode = 'markers',
    name = str(data.iloc[0, data.columns.get_loc('grp')]),
    marker = dict(
        size = '5',
        color = data.iloc[0, data.columns.get_loc('clr')],
    )
    )
    
def scatter_plot_color(data_df):
    """Builds scatter plot traces given a data frame containing x, y and grp
    data

    Args:
        data: A data frame containin the following columns
            x - Data for the x axis
            y - Data for the y axis
            grp - group indicators
    Returns:
        A list of traces to plot with plotly

    """
    
    clrs = np.array(cl.scales['12']['qual']['Paired'])
    
    # Determine unique groups and map them to ints
    unique_grps = np.unique(data_df['grp'])
    num_grps = len(unique_grps)
    grp_nums = np.array(range(num_grps))
    
    if num_grps > 12:
        clrs = cl.to_rgb(cl.interp( clrs, num_grps ))
    
    # Map the given groups to their corresponding ints
    grp_inds = grp_nums[um.match_arrays(np.array(data_df['grp']), unique_grps)]

    # Add column for colour
    plot_data_df = data_df
    plot_data_df['clr'] = clrs[grp_inds]
    
    # Create a scatter plot trace for each group, coloring them differently
    traces = []    
    for i in unique_grps:
        inds = np.where(plot_data_df['grp'] == i)[0]
        trace0 = get_scatter_plot_trace_color_grp(plot_data_df.iloc[inds,:])
        traces.append(trace0)
    
    return traces

def get_line_sensor_trace(sensor_data, y_axis, clr, name, t_axis=None, lines = True, show_leg = True):
    
    if t_axis is None:
        t_data = range(sensor_data.shape[0])
    else:
        t_data = sensor_data[t_axis]
        
    if lines:
        plot_type = 'lines'
    else:
        plot_type = 'markers'
        
    return go.Scatter(
        x = t_data,
        y = sensor_data[y_axis],
        mode = plot_type,
        name = name,
        line = dict(
            color = clr
    
        ),
        showlegend = show_leg
    )

def line_plot_sensor_data(sensor_data_series, y_axis, clr, name, show_leg_plot, 
                          t_axis=None, lines = False):
    traces = []
    cnt = 0
    for x in sensor_data_series:
        if cnt == 0:
            show_leg = True & show_leg_plot
        else:
            show_leg = False & show_leg_plot
        traces.append(get_line_sensor_trace(x, y_axis, clr, name, t_axis, lines, show_leg))
        cnt = cnt + 1
    return traces


def subplot_helper(nrow, ncol, plot_traces, plot_titles=None):
    if plot_titles == None:
        fig = tools.make_subplots(rows = nrow, cols = ncol, print_grid=False)
    else:
        fig = tools.make_subplots(rows = nrow, cols = ncol, subplot_titles = plot_titles, print_grid=False)
        
    rows = np.array(sorted(range(1, nrow + 1) * ncol))
    cols = np.array(range(1, ncol + 1) * nrow)
    
    plot_num = 0
    for pl in plot_traces:
        r = rows[plot_num]
        c = cols[plot_num]
        
        for tr in pl:
            fig.append_trace(tr, r, c)
            
        plot_num += 1
        
    return fig

def subplot_helper_fig(nrow, ncol, plot_figs):
    
    
    plot_traces = [x['data'] for x in plot_figs]
    plot_titles = [x['layout']['title'] for x in plot_figs]
    plot_yranges = [x['layout']['yaxis']['range'] for x in plot_figs]
    plot_xranges = [x['layout']['xaxis']['range'] for x in plot_figs]
    plot_ytitles = [x['layout']['yaxis']['title'] for x in plot_figs]
    plot_ytitlefonts = [x['layout']['yaxis']['titlefont'] for x in plot_figs]
      
    fig = tools.make_subplots(rows = nrow, cols = ncol, subplot_titles = plot_titles, print_grid=False)
    
    rows = np.array(sorted(range(1, nrow + 1) * ncol))
    cols = np.array(range(1, ncol + 1) * nrow)
    
    plot_num = 0
    for pl in plot_traces:
        r = rows[plot_num]
        c = cols[plot_num]
        
        for tr in pl:
            fig.append_trace(tr, r, c)
            
        plot_num += 1
        
    for i in range(len(plot_figs)):
        fig['layout']['yaxis' + str(i+1)].update(range = plot_yranges[i])
        fig['layout']['xaxis' + str(i+1)].update(range = plot_xranges[i])
        fig['layout']['yaxis' + str(i+1)].update(title = plot_ytitles[i])
        fig['layout']['yaxis' + str(i+1)].update(titlefont = plot_ytitlefonts[i])
    
    return fig

def feature_strip_plot2(feats, feat_names, grp_col, grp_colors, ncol,
                        xr = None, yr = None):
    """Creats a lattice of strip plots for the given grouped features

    A strip plot shows the distribution of a feature in a compact way,
    in effectively one dimension.  This function produces a strip plot
    for each feature, grouped by the given grouping column.

    Args:
        feats: A dataframe containing the features (on the columns) and a
            grouping variable.
        feat_names: A list of strings of features to plot, matching the column 
            names in feats.
        grp_col: The name (string) of the column in feats used for grouping.
        grp_colors: A dict containing the colors (rgb strings) for each
            group in the grouping column. (E.g. {'group1':'rgb(228,26,28)'}).
        ncol: The number of columns (int) in the lattice plot.

    Returns:
        A plotly figure.

    """  
        
    grps = np.unique(feats[grp_col])

    if yr == None:
        yr = [-3, 3]

    figs = []
    ind = 0
    for f in feat_names:
    
        layout = go.Layout(
            title = 'feat ' + str(f),
            titlefont = dict(size=10),
            autosize=False,
            yaxis = dict(
                range=[-3, 3],
                titlefont=dict(
                    size=12
                )
                ),
            xaxis = dict(range = xr)
            )
                
        if ind == 0:
            show_leg = True
        else:
            show_leg = False
            
        traces = []
        for g in grps:
            data = feats.loc[feats[grp_col] == g, f]
            plt = go.Scatter(
                    x = list(data), 
                  y = [random.uniform(-1, 1) for x in range(data.shape[0])],
                  mode = 'markers', name = g, 
                  marker = Marker(color=grp_colors[g]),
                  showlegend = show_leg)
            traces.append(plt)
        
        fig = go.Figure(data=traces, layout=layout)
        figs.append(fig)
        ind = ind + 1
    
    nrow = int(np.ceil(float(len(feat_names)) / float(ncol)))
    
    return subplot_helper_fig(nrow, ncol, figs)


def feature_distribution_plot(feats, feat_names, grp_col, grp_colors, ncol,
                              plot_ranges = None, dens_est = False,
                              dens_num = 100, title_font = 12, all_show_leg = True,
                              lattice = True, titles = None, highlight_samps = [],
                              hl_clr = 'black'):
    """Creats a lattice of histogram plots for the given grouped features

    Args:
        feats: A dataframe containing the features (on the columns) and a
            grouping variable.
        feat_names: A list of strings of features to plot, matching the column 
            names in feats.
        grp_col: The name (string) of the column in feats used for grouping.
        grp_colors: A dict containing the colors (rgb strings) for each
            group in the grouping column. (E.g. {'group1':'rgb(228,26,28)'}).
        ncol: The number of columns (int) in the lattice plot.

    Returns:
        A plotly figure.

    """ 
    grps = np.unique(feats[grp_col])

    figs = []
    cnt = 0
    for f in feat_names:

        if cnt == 0:
            show_leg = True & all_show_leg
        else:
            show_leg = False & all_show_leg
        
        if titles == None:
            t = f
        else:
            t = titles[f]
        
        if plot_ranges != None:            
            layout = go.Layout(
                title = t,
                titlefont = dict(size=10),
                autosize=False,
                xaxis = dict(range = plot_ranges[f]))
        else:
            layout = go.Layout(
                title = t,
                titlefont = dict(size=10),
                autosize=False)
            
        traces = []
        if plot_ranges != None:
            f_min = plot_ranges[f][0]
            f_max = plot_ranges[f][1]
        else:
            f_min = 0.7 * min(feats.loc[:, f])
            f_max = 1.3 * max(feats.loc[:, f])
        xpts = np.linspace(f_min, f_max, dens_num)
        
        for g in grps:
            
            data = feats.loc[feats[grp_col] == g, f]
            y = np.array(data)
            y = y[np.isnan(y) == False]            
            #if (dens_est) & (len(y) > 0) &\
            #(len(np.unique(y)) > 3*len(y) / 4):
            if (dens_est) & (len(y) > 0):    
                kde = KernelDensity(kernel='gaussian', 
                                    bandwidth=0.2).fit(y[:, np.newaxis])
                log_dens = kde.score_samples(xpts[:, np.newaxis])
                plt = go.Scatter(x=xpts, y=np.exp(log_dens),
                                 mode='lines',
                                 line=dict(color = grp_colors[g], width=2),
                                 name = g, showlegend = show_leg) 
                traces.append(plt)

            else:
                if len(y) > 0:
                    plt = go.Histogram(x = list(data), 
                          marker=Marker(color=grp_colors[g]),
                          name = g, showlegend = show_leg)                
                    
                    traces.append(plt)
                    
        if len(highlight_samps) > 0:
            for hs in highlight_samps:
                plt = go.Scatter(x = [feats.loc[hs,f]] * 2,
                                 y = [0.0, 1.0],
                                 line = dict(color=hl_clr, width = 2),
                                 mode = 'lines')
                traces.append(plt)

        
        fig = go.Figure(data=traces, layout = layout)
        fig['layout'].update(titlefont = dict(size=title_font))
        figs.append(fig)
        cnt = cnt + 1
    
    if lattice:
        nrow = int(np.ceil(float(len(feat_names)) / float(ncol)))
        
        return subplot_helper_fig(nrow, ncol, figs) 
    
    return figs

def feature_distribution_plot_mult_modes(feats, feat_names, grp_col, grp_colors, 
                                         grp_modes, ncol, plot_ranges = None,
                                         dens_num = 100):
    """Creats a lattice of histogram plots for the given grouped features,
    where each group can have a different plot type.

    Args:
        feats: A dataframe containing the features (on the columns) and a
            grouping variable.
        feat_names: A list of strings of features to plot, matching the column 
            names in feats.
        grp_col: The name (string) of the column in feats used for grouping.
        grp_colors: A dict containing the colors (rgb strings) for each
            group in the grouping column. (E.g. {'group1':'rgb(228,26,28)'}).
        grp_modes: A dict containing the plotting for each group.  Each mode
            must be one of the following:
                'hist' - A histogram
                'dens' - A fitted density histogram
                'markers' - Scatter plot
        ncol: The number of columns (int) in the lattice plot.

    Returns:
        A plotly figure.

    """ 
    
    grps = np.unique(feats[grp_col])

    figs = []
    cnt = 0
    for f in feat_names:

        if cnt == 0:
            show_leg = True
        else:
            show_leg = False
        
        if plot_ranges != None:            
            layout = go.Layout(
                title = f,
                titlefont = dict(size=10),
                autosize=False,
                xaxis = dict(range = plot_ranges[f]))
        else:
            layout = go.Layout(
                title = f,
                titlefont = dict(size=10),
                autosize=False)
            
        traces = []
        if plot_ranges != None:
            f_min = plot_ranges[f][0]
            f_max = plot_ranges[f][1]
        else:
            f_min = min(feats.loc[:, f])
            f_max = max(feats.loc[:, f])
        xpts = np.linspace(f_min, f_max, dens_num)        
        for g in grps:
            
            data = feats.loc[feats[grp_col] == g, f]
            
            if grp_modes[g] == 'dens':
                y = np.array(data)
                y = y[np.isnan(y) == False]
                kde = KernelDensity(kernel='gaussian', 
                                    bandwidth=0.2).fit(y[:, np.newaxis])
                log_dens = kde.score_samples(xpts[:, np.newaxis])
                plt = go.Scatter(x=xpts, y=np.exp(log_dens),
                                 mode='lines',
                                 line=dict(color = grp_colors[g], width=2),
                                 name = g, showlegend = show_leg)  
            elif grp_modes[g] == 'markers':
                plt = go.Scatter(x = list(data), y = [1.0] * len(data),
                                 mode='markers',
                                 marker = dict(color = grp_colors[g], size = 5),
                                 name = g, showlegend = show_leg)                  
            else:
                plt = go.Histogram(x = list(data), 
                      marker=Marker(color=grp_colors[g]),
                      name = g, showlegend = show_leg)
                
            traces.append(plt)

        
        fig = go.Figure(data=traces, layout = layout)
        figs.append(fig)
        cnt = cnt + 1
    
    nrow = int(np.ceil(float(len(feat_names)) / float(ncol)))
    
    return subplot_helper_fig(nrow, ncol, figs)     


def feature_distribution_plot_users(feats, users, feat_names, 
                                    grp_col, grp_colors, dens_est = False,
                                    dens_num = 50):
    """Creates a lattice of histogram plots for grouped features for many users.
    
    Each column contains the histogram feature plots for a single user.

    Args:
        feats: A dictionary containing the features (on the columns) and a
            grouping variable for each user.
        users: A subset of the keys of 'feats' to plot
        feat_names: A list of strings of features to plot, matching the column 
            names in feats.
        grp_col: The name (string) of the column in feats used for grouping.
        grp_colors: A dict containing the colors (rgb strings) for each
            group in the grouping column. (E.g. {'group1':'rgb(228,26,28)'}).
        ncol: The number of columns (int) in the lattice plot.
        dens_est: A boolean indicating whether or not to show density
            estimation plot instead of histogram.

    Returns:
        A plotly figure.

    """ 
    
    grps = np.unique(feats[users[0]][grp_col])

    

    figs = []
    cnt = 0
    user_plt_cnt = 0
    for f in feat_names:

        vals = []
        for u in users:
            vals.extend(feats[u][f])
        fmin = np.nanpercentile(vals, 2)
        fmax = np.nanpercentile(vals, 98)
        #fmin = min([min(feats[u][f]) for u in users])
        #fmax = max([max(feats[u][f]) for u in users])
        
        feat_plot_cnt = 0
        for u in users:
        
            if cnt == 0:
                show_leg = True
            else:
                show_leg = False
            
            if user_plt_cnt == 0:
                plt_title = u
            else:
                plt_title = ''
                
            if feat_plot_cnt == 0:
                y_title = f
            else:
                y_title = ''
            
            layout = go.Layout(
                title = plt_title,
                titlefont = dict(size=10),
                yaxis = dict(title = y_title,
                           titlefont = dict(size=18),
                           color='black'),
                xaxis = dict(range = [fmin, fmax]),        
                autosize=False)
            
            f_min = min(feats[u].loc[:, f])
            f_max = max(feats[u].loc[:, f])
            xpts = np.linspace(f_min, f_max, dens_num)
            traces = []
            for g in grps:

                data = feats[u].loc[feats[u][grp_col] == g, f]                
                if dens_est:
                    y = np.array(data)
                    y = y[np.isnan(y) == False]
                    kde = KernelDensity(kernel='gaussian', 
                                        bandwidth=0.75).fit(y[:, np.newaxis])
                    log_dens = kde.score_samples(xpts[:, np.newaxis])
                    plt = go.Scatter(x=xpts, y=np.exp(log_dens),
                                     mode='lines',
                                     line=dict(color = grp_colors[g], width=2),
                                     name = g, showlegend = show_leg)
                else:

                    plt = go.Histogram(x = list(data), 
                          marker=Marker(color=grp_colors[g]),
                          name = g, showlegend = show_leg)
                traces.append(plt)
            
            fig = go.Figure(data=traces, layout = layout)
            figs.append(fig)
            cnt = cnt + 1
            feat_plot_cnt = feat_plot_cnt + 1
        user_plt_cnt = user_plt_cnt + 1
    
    ncol = len(users)
    nrow = len(feat_names)
    
    return subplot_helper_fig(nrow, ncol, figs) 

def feature_summary_heatmap(feat_values, zmin = None, zmax = None):      
    """Creats a heatmap of feature summary values (like feature important)
    
    The features are on the rows and users on the columns of the heatmap.
    (There are usually more features and users)  

    Args:
        feat_values: A dataframe where the columns are the features and
            the users are the indices.
        zmin: An optional value for min of the z axis
        zmax: An optional value for max of the z axis


    Returns:
        A plotly figure.
        
    """
    
    f_names = feat_values.columns
    traces = [go.Heatmap(z = [list(feat_values[f]) for f in f_names],
                         x = feat_values.index,
                         y = f_names,
                         zmin = zmin,
                         zmax = zmax)]
    
    return go.Figure(data = traces, 
                     layout = go.Layout())
    
def features_tsne(feats, feat_names, grp_col, grp_colors, title = '',
                  lrate = 200.0, niter = 1000, save_res = [],
                  show_pt_nums = False):
    """Creats a T-SNE plot for features given grouping column

    Args:
        feats: A dataframe where the columns are the features and
            the rows are the samples.
        feat_names: A list containing the features to transform.
        grp_col: The column name in feats where the grouping indices are.
        grp_colors: A dictionary indicating the color to use for each group
        title: An optional plot title

    Returns:
        A plotly figure.
        
    """ 
    
    t_sne = TSNE(n_components=2, random_state=0,
                 learning_rate = lrate, n_iter = niter)
    feats_2d = t_sne.fit_transform(np.array(feats.loc[:, feat_names]))
    
    grps = np.unique(feats[grp_col])
    
    traces = []
    for g in grps:
        
        inds = np.array(feats[grp_col]) == g
        if show_pt_nums:
            txt = np.where(inds)[0]        
        
            traces.append(go.Scatter(x = feats_2d[inds, 0], 
                                     y = feats_2d[inds, 1],
                                     text = txt,
                                     name = g,
                                     mode = 'markers+text',
                                     marker=Marker(color=grp_colors[g])))
        else:
            traces.append(go.Scatter(x = feats_2d[inds, 0], 
                                     y = feats_2d[inds, 1],
                                     name = g,
                                     mode = 'markers',
                                     marker=Marker(color=grp_colors[g])))            
        
    fig = go.Figure(data = traces, layout = go.Layout(title = title))
    save_res.append(feats_2d)
    return fig

def pairwise_feat_plots(feats, feat_names, grp_col, grp_colors, nrow,
                        xr = None, yr = None):
    """Creats plots where each one has one feature vs. another.
    
    Features are paired up so no feature is shown twice, except when
    there is an odd-number of features. (in that case, one feature will
    be plotted twice)  This is more compact than the more common pairwise
    plotted where every feature is plotted against every other.

    Args:
        feats: A dataframe where the columns are the features and
            the rows are the samples.
        feat_names: A list containing the features to transform.
        grp_col: The column name in feats where the grouping indices are.
        grp_colors: A dictionary indicating the color to use for each group
        title: An optional plot title

    Returns:
        A plotly figure.
        
    """     
    
    mid_pt = int(np.ceil(float(len(feat_names)) / 2.0))
    
    f_grp1 = feat_names[0:mid_pt]
    f_grp2 = feat_names[mid_pt:len(feat_names)]
    
    if len(feat_names) % 2 != 0:
        f_grp2.append(f_grp2[0])
    
    feat_groups = zip(f_grp1, f_grp2)
    
    grps = np.unique(feats[grp_col])
    
    figs = []
    ind = 0
    for f_g in feat_groups:
        traces = []
       
        if ind == 0:
            show_leg = True
        else:
            show_leg = False
   
        for g in grps:
            traces.append(
                    go.Scatter(x = feats.loc[feats[grp_col] == g, f_g[0]], 
                               y = feats.loc[feats[grp_col] == g, f_g[1]],
                               name = g,
                               mode = 'markers',
                               marker = Marker(color=grp_colors[g]),
                               showlegend = show_leg))

        figs.append(go.Figure(
                data = traces,
                layout = go.Layout(xaxis = dict(range = xr),
                                   yaxis = dict(range = yr),
                                   title = str(f_g[0]) + ' vs ' + str(f_g[1]))))
        ind = ind + 1

    ncol = int(np.ceil(float(len(feat_groups)) / float(nrow)))
    
    return subplot_helper_fig(nrow, ncol, figs)    

def roc_curve_uar_aar(scores, labels, user_lab, att_lab, title, thresholds,
                      roc_res, show_chance = True,
                      bigger_score_better = True):

    # Set labels to correct range
    arr_labels = np.array(labels)
    gr_truth = np.array(labels)
    gr_truth[arr_labels == user_lab] = 1
    gr_truth[arr_labels == att_lab] = -1    
    
    arr_scores = np.array(scores)
    
    # Get list of thresholds
    if len(thresholds) == 0:
        fpr, tpr, new_thresholds = roc_curve(gr_truth, scores)

        for t in new_thresholds:
            thresholds.append(t)

    # Get uar and aar
    uar = []
    aar = []
    for t in thresholds:
        uar_new, aar_new = mam.uar_aar_score_cond(arr_scores, t, arr_labels, 
                                                  user_lab, att_lab,
                                                  bigger_score_better)
        uar.append(uar_new)
        aar.append(aar_new)
        
    auc_val = auc(aar, uar)
        
    traces = [go.Scatter(x = aar, y = uar,
                         name = 'ROC, AUC = ' + '%.3f' % auc_val,
                         mode = 'lines',
                         marker = dict(color = 'blue'))]
    
    # Add chance line
    if show_chance:
        traces.append(go.Scatter(x = np.arange(0.0, 1.1, 0.1),
                                 y = np.arange(0.0, 1.1, 0.1),
                                 name = 'Chance',
                                 mode = 'lines',
                                 marker = dict(color = 'orange')))    
    
    roc_res['thresh'] = thresholds
    roc_res['uar'] = uar
    roc_res['aar'] = aar
    roc_res['auc'] = auc_val
    
    # Get optimal UAR/AAR
    m_uar_aar = np.array([np.mean([x[0], 1.0 - x[1]]) for x in zip(uar, aar)])
    
    max_ind = np.argmax(m_uar_aar)
    
    opt_res = {'optimal_thresh': np.array(thresholds)[max_ind],
                     'optimal_uar': np.array(uar)[max_ind],
                     'optimal_aar': np.array(aar)[max_ind],
                     'auc': auc_val}
    
    roc_res['opt_res'] = opt_res
    
    
    return go.Figure(data = traces, 
                     layout = go.Layout(title = title,
                                        xaxis = dict(title = 'aar'),
                                        yaxis = dict(title = 'uar')))
    
    
def score_distribution_plot(scores, train_score_key, test_score_key, 
                            test_conds_key, uar_thresh, unames):
    good_uar_val = 0.9

    train_test_score_plts = {}
    cnt = 0
    for u in unames:
        if cnt == 0:
            show_leg = True
        else:
            show_leg = False
        
        traces = []
        sc = np.array(scores[u][test_score_key])
        cnd = np.array(scores[u][test_conds_key])
        
        # Get threshold value at uar cutoff
        thresh_good_uar = mam.thresh_at_uar(
                np.array(scores[u][test_score_key]), 
                np.array(scores[u][test_conds_key]), 1, -1, uar_thresh)
        if thresh_good_uar == 0.0:
            thresh_good_uar = -1.0
        
        traces.append(go.Histogram(x = sc[cnd == 1], 
                                   marker = dict(color = 'blue'), name = 'user',
                                   showlegend = show_leg))
        traces.append(go.Histogram(x = sc[cnd == -1], 
                                   marker = dict(color = 'red'), name = 'attacker',
                                   showlegend = show_leg))
        traces.append(go.Histogram(x = scores[u][train_score_key], 
                                   marker = dict(color = 'black'), name = 'model',
                                   showlegend = show_leg))    
        traces.append(go.Scatter(x = [thresh_good_uar, thresh_good_uar, thresh_good_uar],
                                y = [0, 15, 30], 
                                text = ['', '%.2f' % thresh_good_uar, ''],
                                name = 'Threshold at ' + str(int(good_uar_val*100.0)) + '% UAR',
                                mode = 'lines+text', marker = dict(color = 'grey'),
                                textposition='right',
                                showlegend = show_leg))
        
        train_test_score_plts[u] = go.Figure(data = traces, 
                                       layout = go.Layout(xaxis = dict(range = [-2.0, 100.0]),
                                                          yaxis = dict(range = [0, 30]),
                                                         title = 'Train vs Test Scores - ' + u))
        cnt = cnt + 1
    return train_test_score_plts
   