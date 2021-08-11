# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

### import key libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
import pytz
from copy import copy, deepcopy

from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import geopandas as gpd
import pyproj

from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)

def visualize_simpleplot(df):
    import contextily as ctx
    gdf = gpd.GeoDataFrame(df, 
                           geometry=gpd.points_from_xy(df[lonColName],
                                                       df[latColName]),
                           crs= {"init": "epsg:4326"}).to_crs(epsg=3857)
    fig,ax = plt.subplots(figsize=(15,10))
    xmin, ymin, xmax, ymax = gdf.total_bounds
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, 
                                     source=ctx.providers.CartoDB.Voyager)
    gdf.plot(ax=ax, zorder=2)
    ax.imshow(basemap, extent=extent, zorder=1)
    plt.show()

def visualize_boundarymap(boundary):
    import contextily as ctx
    fig,ax = plt.subplots(figsize=(15,10))
    x1, y1 = (boundary[0], boundary[1])
    x2, y2 = (boundary[2], boundary[3])
    basemap, extent = ctx.bounds2img(x1, y1, x2, y2, ll=True,
                                     source=ctx.providers.CartoDB.Voyager)
    ax.imshow(basemap, extent=extent)
    plt.show()
    
    
def plot_density_map(latitudes, longitudes, center, bins, radius, ax=None, annotations=None):
    '''
    Parameters
    ----------
    latitudes, longitudes : array-like
        Array contaning the lat and lon coordinates of each user on a selected day.
    center : tuple-like
        The (lat, lon) of the center where to compute the population density around.
    bins : int
        The number of bins to use horizontally and vertically in the region around the
        center.
    radius : float
        The space to consider above, below, left and right of the center (same unity
        of the center).
    ax : matplotlib.axes
        The axes to use. If `None` a new figure will be created.
    annotations : dict
        A dictionary of annotations to be put on the map in the form of
        `{"Text": {kwargs to ax.annotate} }`. Will be used as `ax.annotate(key, **value)`.
    
    Returns
    -------
    res, ax
        The putput of `ax.hist2d` and the axis itself.
    '''
    
    cmap = copy(plt.cm.jet)
    cmap.set_bad((0,0,0))  # Fill background with black

    # Center the map around the provided center coordinates
    histogram_range = [
        [center[1] - radius, center[1] + radius],
        [center[0] - radius, center[0] + radius]
    ]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_aspect("equal")
        
    res = ax.hist2d(longitudes, latitudes, bins=bins, norm=LogNorm(),
                       cmap=cmap, range=histogram_range)
        
    if annotations:
        for name, annot in annotations.items():
            ax.annotate(name, **annot)
    ax.axis('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid('off')
    ax.axis('off')
    return res, ax
    
import matplotlib.colors as mcolors
import mpl_toolkits
import matplotlib.gridspec as gridspec
def shori_density_map(data, xbins, ybins, ax=None, annotations=None, vmin=-2, vmax=2):
    '''
    Parameters
    ----------
    data : array-like
        Array contaning the raster of the population density to be plot.
    xbins, ybins : array-like
        The bins used to construct the raster. Will be used to limit the plot area.
    ax : matplotlib.axes
        The axes to use. If `None` a new figure will be created.
    annotations : dict
        A dictionary of annotations to be put on the map in the form of
        `{"Text": {kwargs to ax.annotate} }`. Will be used as `ax.annotate(key, **value)`.
    vmin, vmax : float
        The values to be passed to the colormap.
    
    Returns
    -------
    res
        The output of `ax.imshow`.
    '''
    
    divnorm = mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax)
    cmap = copy(plt.cm.jet)
    diff = np.ma.masked_where(data == 0, data)
    cmap.set_bad(color='black')  # Fill background with black
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        # ax.set_aspect("equal")
    else:
        fig = ax.figure
        
    res = ax.imshow(diff.T, cmap=cmap,
                    extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]], 
                    norm=divnorm, origin="lower")
    # divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    # cax = divider.append_axes('right', '5%', pad='3%')
    # fig.colorbar(res, cax=cax)
    
    if annotations is not None:
        for name, kwar in annotations.items():
            ax.annotate(name, **kwar)
    # ax.axis('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid('off')
    ax.axis('off')
    
    return res


def plot_pop(df, title, empiric_pop="POBTOT", data_pop="POP_HFLB", alpha=.1, verbose=True):
    '''
    Plot the scatter-plot between empiric_pop and data_pop columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        As dataframe containing the two columns selected.
    '''

    plt.scatter(df[empiric_pop], df[data_pop], alpha=alpha)
    plt.loglog();

    plt.xlim(1, df[empiric_pop].max()*2.)
    plt.ylim(1, df[data_pop].max()*2.)

    lr = LinearRegression(fit_intercept=False)

    valid = df[(~df[empiric_pop].isna())
                          & (~df[data_pop].isna())]

    X = valid[empiric_pop].values
    Y = valid[data_pop].values

    idxs = np.logical_and(X>0,Y>0)
    X = X[idxs]
    Y = Y[idxs]

    lr.fit(X.reshape(-1,1),
           Y.reshape(-1,1))

    X_pred = np.array(sorted(X)).reshape(-1,1)
    Y_pred = lr.predict(X_pred)

    import statsmodels.api as sm
    from scipy import stats

    X2 = sm.add_constant(np.log10(X))
    est = sm.OLS(np.log10(Y), X2)
    est2 = est.fit()

    if verbose:
        print(est2.summary())

    plt.plot(X_pred, Y_pred, "--C1", label="Slope: %.02f"
             % (est2.params[1]))
    
    plt.title(title)

    plt.legend()
    plt.xlabel(empiric_pop + title)
    plt.ylabel(data_pop)

    return est2
