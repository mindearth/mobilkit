# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

import functools
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, correspond, inconsistent
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import calinski_harabasz_score
from requests import get as r_get

from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)


# def fudf(val):
#     '''Deprecated function to flatten a column of sets.
#     '''
#     return functools.reduce(lambda x, y: list( set(x).union(set(y)) ), val)
# flattenSetsUdf = sqlF.udf(fudf, ArrayType(StringType()))


def computeClusters(results, signal, metric='cosine', nClusters=[2,3,4,5,6,7,8,9]):
    '''Function to compute the clusters.

    Parameters
    ----------
    results : dict
        As returned from :attr:`mobilkit.spatial.computeResiduals`.
    signal : str
        One of ``'mean', 'zscore', 'residual'``, the indicator to use to cluster the areas.
    metric : str, optional
        One of ``'cosine', 'euclidean'``, the metric used to compute the linkage matrix. Default to ``cosine``.
    nClusters : list, optinal
        The list or set of number of clusters to try.

    Returns
    -------
    results_clusters : dict
        A dict with the results to be used for plotting and inspection.
    '''
    assert signal in ["mean", "zscore", "residual"]
    assert metric in ["cosine", "euclidean"]

    nClusters_hierarchical = np.array([int(c) for c in nClusters])

    vecsAsArrayNorm = results[signal].copy()
    zeri = np.isclose(np.abs(vecsAsArrayNorm).sum(axis=1), 0, rtol=1e-6)
    minimo = vecsAsArrayNorm[np.abs(vecsAsArrayNorm)!=0].min()
    vecsAsArrayNorm[zeri,:] = minimo

    models = []
    inertia = []
    explained = []
    explainedCosine = []

    scores = []
    inconsistents = []
    ks = []

    vectorsTMP = vecsAsArrayNorm
    distanceMatrix = pdist(vectorsTMP, metric=metric)
    linkagesMatrix = linkage(distanceMatrix, method='ward')

    coph_coeff, coph_dists = 0, 0

    for nCenters in nClusters_hierarchical:
        labels = fcluster(linkagesMatrix, nCenters, criterion="maxclust")
        ks.append(nCenters)
        scores.append(calinski_harabasz_score(vectorsTMP, labels))
        inconsistents.append(inconsistent(linkagesMatrix, d=nCenters))
        print("Done n clusters = %02d" % nCenters)

    results_clusters = {
        "vecs": vecsAsArrayNorm,
        "signal": signal,
        "ks": ks,
        "models": models,
        "inconsistents": inconsistents,
        "scores": scores,
        "metric": metric,
        "distanceMatrix": distanceMatrix,
        "linkagesMatrix": linkagesMatrix,
            }
    return results_clusters

def checkScore(results_clusters, score="scores"):
    '''
    Function to plot the score of clustering.
    
    Parameters
    ----------
    results_clusters : dict
        As returned by :attr:`mobilkit.tools.computeClusters`.

    score : str, optional
        One of ``"scores", "inconsistents"``. If ``scores``, the best split is at a local maximum of the score.

    Returns
    -------
    ax : axis
        The ax of the figure.
    '''
    assert score in ["scores", "inconsistents"]

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    iii_series = 0

    ax.plot(results_clusters["ks"], results_clusters[score],
             "o--C%d" % iii_series, lw=2, ms=10, label=results_clusters["metric"])

    ax.set_xlabel("n clusters", size=16)
    ax.set_ylabel("[calinski_harabaz_score]" if score=="scores" else "[inconsistents]", size=16)
    ax.legend(loc=1)

    plt.tight_layout()

    return ax

def visualizeClustersProfiles(results_clusters, nClusts=5,
                              showMean=False,
                              showMedian=True,
                              showCurves=True,
                              together=False):
    '''Function to plot the temporal profiles of clustering.

    Parameters
    ----------
    results_clusters : dict
        As returned by :attr:`mobilkit.tools.computeClusters`.
    nClusts : int, optional
        The number of clusters to use.
    showMean : bool, optional
        Whether or not to show the mean curve of the cluster.
    showMedian : bool, optional
        Whether or not to show the median curve of the cluster.
        If both median and mean are to be plotted, only the median will be shown.
    showCurves : bool, optional
        Whether or not to show the curves of the cluster in transparency.
    together : bool, optional
        Whether to plot all the cluster in one plot. Default ``False``.

    Returns
    -------
    ax : axis
        The ax of the figure.
    '''
    vecsAsArrayNorm = results_clusters["vecs"]
    linkagesMatrix = results_clusters["linkagesMatrix"]

    labels = fcluster(linkagesMatrix, nClusts, criterion="maxclust")

    if not together:
        fig, axs = plt.subplots(nClusts, 1, figsize=(12,4*nClusts))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12,4))
    iii_series = 0

    for label in sorted(np.unique(labels)):
        if together:
            ax = axs
        else:
            ax = axs[label-1]
        tmp_data = vecsAsArrayNorm[labels==label]

        if showCurves:
            ax.plot(tmp_data.T, alpha=.1, color="C%d"%(label-1))
            ax.plot(np.median(tmp_data,axis=0), alpha=1, color="C%d"%(label-1), label="Cluster %d"%label)

        if showCurves:
            medianColor = "k"
        else:
            medianColor = "C%d" % (label-1)

        if showMedian:
            ax.plot(np.median(tmp_data,axis=0), alpha=1, color="C%d"%(label-1), label="Median %d"%label)
            ax.plot(np.median(tmp_data,axis=0), alpha=1, color=medianColor, lw=3)
        elif showMean:
            ax.plot(np.mean(tmp_data,axis=0), alpha=1, color="C%d"%(label-1), label="Mean %d"%label)
            ax.plot(np.mean(tmp_data,axis=0), alpha=1, color=medianColor, lw=3)

        # ax.set_xlim(0,24-1)
        # ax.set_ylim(-2,2)
        # ax.set_xticks(range(0,24,2))
        # ax.set_xticklabels(["%02d:00"%h for h in range(0,24,2)], size=14)
        ax.yaxis.set_tick_params(labelsize=14)

        ax.set_xlabel("Hour", size=18)
        ax.set_ylabel(results_clusters["signal"], size=18)

        if not together: ax.legend(loc="upper center", fontsize=14)

    if together: ax.legend(loc="upper center", fontsize=14)
    fig.tight_layout()

    return ax


def plotCommunities(results_clusters, nClusts):
    '''
    Function to plot the similarity matrix between areas.
    
    Parameters
    ----------
    results_clusters : dict
        As returned by :attr:`mobilkit.tools.computeClusters`.
    nClusts : int, optional
        The number of clusters to use.

    Returns
    -------
    ax : axis
        The ax of the figure.
    '''
    # Distance matrix
    fig, ax = plt.subplots(1,1,figsize=(14,12))

    distanceMatrix = results_clusters["distanceMatrix"]
    linkagesMatrix = results_clusters["linkagesMatrix"]
    labels = fcluster(linkagesMatrix, nClusts, criterion="maxclust")
    labels_sorted = np.argsort(labels)

    distanceMatrixSorted = squareform(distanceMatrix)
    distanceMatrixSorted = distanceMatrixSorted[labels_sorted,:]
    distanceMatrixSorted = distanceMatrixSorted[:,labels_sorted]

    NpInClust = Counter(labels)
    cum = 0
    for l, c in sorted(NpInClust.items()):
        plt.plot([cum,cum+c,cum+c,cum,cum], [cum,cum,cum+c,cum+c,cum], "-C1", lw=5)
        cum += c

    plt.xticks([])
    plt.yticks([])

    plt.imshow(distanceMatrixSorted, cmap="viridis_r")
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=20)
    cbr.ax.set_ylabel("%s distance" % results_clusters["metric"].title(), size=24)

    return ax


def plotClustersMap(gdf, results_clusters, mappings, nClusts=5):
    '''Function to plot the similarity matrix between areas.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame with at least the ``tile_ID`` and ``geometry`` columns.
    results_clusters : dict
        As returned by :attr:`mobilkit.tools.computeClusters`.
    mappings : dict
        The mappings as returned by :attr:`mobilkit.spatial.computeResiduals`.
    nClusts : int, optional
        The number of clusters to use.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        The original GeoDataFrame with an additional column ``'cluster'`` containing the cluster assigned to that area.
    ax : axis
        The ax of the figure.
    '''
    # Visualize them on map
    # Create a column on the gdf with the cluster label
    linkagesMatrix = results_clusters["linkagesMatrix"]
    labels = fcluster(linkagesMatrix, nClusts, criterion="maxclust")

    gdf["cluster"] = gdf["tile_ID"].apply(lambda z:
                labels[mappings["area2idx"][z]]
                if z in mappings["area2idx"] else None)

    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.set_aspect("equal")


    for label in sorted(np.unique(labels)):
        gdf[gdf["cluster"] == label].plot(ax=ax,color="C%d"%(label-1))

    selected_areas = set([k for k in mappings["area2idx"].keys()])
    gdf[gdf[zidColName].isin(selected_areas)]\
                        .plot(color="none", edgecolor="black", ax=ax, zorder=1)

    return gdf, ax


def computeGDFbounds(gdf: gpd.GeoDataFrame) -> dict:
    '''
    Computes the bounds of a `geopandas.GeoDataFrame` `gdf`
    and returns a dictionary with the `min(x,y)` and `max(x,y)`
    keys and their value as values (minx is the minimum longitude).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame with at least the ``geometry`` column.

    Returns
    -------
    bounds : dict
        The mapping between the min/max x and y and their values.
    '''
    return gdf.bounds.agg({
            'minx': min,
            'miny': min,
            'maxx': max,
            'maxy': max,
        }).to_dict()


def osrm_time_trip(latlon_start,
                    latlon_end,
                    osrm_url,
                    what='duration',
                    max_trip_duration=10800,
                    max_trip_distance=100.):
    '''
    Queries an OSRM backend server on `osrm_url` for the travel time or distance (depending
    on `what`) between `latlon_start` and `latlon_end`.

    Parameters
    ----------
    latlon_start, latlon_end : list or np.array
        The latitude and longitude of the initial and final trip point (in WGS84, EPSG:4326 crs).
    osrm_url : str
        A valid table endpoint url of a OSRM-bakend service (like `http://localhost:5000/table/v1/car/`).
    what : str, optional
        Whether you want the `duration` or `distance` or `duration,distance`
        of the trip.
    max_trip_duration : float, optional
        The maximum trip duration in seconds that will be put in place of NaNs or invalid trips.
    max_trip_distance : float, optional
        The maximum trip distance in km that will be put in place of NaNs or invalid trips.
    Returns
    -------
    duration : float or tuple
        The duration (length) of the trip in seconds (meters) or the two (duration,distance) if
        `what==duration,distance`.
    
    '''
    # Port to meters
    max_trip_distance = max_trip_distance*1000.
    assert what in ['duration', 'distance', 'duration,distance']
    doubleReturn = what == 'duration,distance'

    if not osrm_url.endswith('/'):
        osrm_url = osrm_url + '/'

    point_i = [latlon_start[1], latlon_start[0]]
    point_f = [latlon_end[1], latlon_end[0]]
    positions_str = ";".join([f'{p[0]},{p[1]}' for p in [point_i, point_f]])

    result = r_get(osrm_url
                   + positions_str
                   + "?sources=0&destinations=1&annotations=%s" % what).json()
    if result["code"] != "Ok":
        print("\nWARNING, missed query at url", osrm_url,
              "with latlon start end", latlon_start, latlon_end)
        tmp_durations = np.ones(1) * max_trip_duration
        tmp_distances = np.ones(1) * max_trip_distance
    else:
        if what == 'duration' or doubleReturn:
            tmp_duration = result['durations'][0][0]
            if tmp_duration is not None and not np.isfinite(tmp_duration):
                tmp_duration = max_trip_duration
        if what == 'distance' or doubleReturn:
            tmp_distance = result['distances'][0][0]
            if tmp_distance is not None and not np.isfinite(tmp_distance):
                tmp_distance = max_trip_distance
            
    if what == 'duration':
        return tmp_duration
    elif what == 'distance':
        return tmp_distance
    elif doubleReturn:
        return (tmp_duration, tmp_distance)
    else:
        raise RuntimeError('SHOULD NOT HAPPEN')

def userHomeWorkTravelTimeOSRM(r, osrm_url,
                               direction='hw',
                               what='duration',
                               max_trip_duration_h=4,
                               max_trip_distance_km=150,
                               ):
    '''
    Queries an OSRM backend server on `osrm_url` for the travel time or distance (depending
    on `what`) between the home and work location for the user in the row `r`.

    Parameters
    ----------
    r : dict or row of a pd.Dataframe
        The row with the home and work latitude and longitude (in WGS84, EPSG:4326 crs)
        in the `lat_home` format, as returned by :attr:`mobilkit.stats.userHomeWorkLocation`.
    osrm_url : str
        A valid table endpoint url of a OSRM-bakend service (like `http://localhost:5000/table/v1/car/`).
    direction : str, optional
        Whether you want the `home->work` ("hw") or `work->home` ("wh") time of the trip.
    what : str, optional
        Whether you want the `duration`, `distance` or `duration,distance` of the trip.
    max_trip_duration_h : float, optional
        The maximum trip duration in hours (or km if asking for distance) that will be put in place
        of NaNs or invalid trips.
    Returns
    -------
    duration : float
        The duration (length) of the trip in seconds (meters) or the two (duration,distance) if
        `what==duration,distance`.
    '''
    assert what in ['duration', 'distance', 'duration,distance']

    if pd.isna(r[lonColName+'_home']) or pd.isna(r[lonColName+'_work']):
        if what == 'duration,distance':
            return [None, None]
        else:
            return None
    else:
        home_place = [r[latColName+'_home'], r[lonColName+'_home']]
        work_place = [r[latColName+'_work'], r[lonColName+'_work']]
        if direction == 'hw':
            p_i = home_place
            p_f = work_place
        elif direction == 'wh':
            p_i = work_place
            p_f = home_place
        else:
            raise RuntimeError('Unknown direction %s'%direction)
            
        
        res = osrm_time_trip(latlon_start=p_i,
                              latlon_end=p_f,
                              osrm_url=osrm_url,
                              what=what,
                              max_trip_duration=max_trip_duration_h*3600,
                              max_trip_distance=max_trip_distance_km,
                              )
        return res