# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

'''Tools and functions to spatially analyze the data.


.. note::
    When determining the home location of a user, please consider that some data providers, like *Cuebiq*, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.

'''
from dask import dataframe as dd
from dask import array as da
import dask.bag as db

from mobilkit.dask_schemas import nunique

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from copy import copy, deepcopy
import geopandas as gpd
from scipy.spatial import Voronoi
import shapely
from sklearn import cluster
from sklearn.metrics.pairwise import haversine_distances

from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import chain
from shapely.geometry import Polygon

from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean


def tessellate(df, tesselation_shp, filterAreas=False, partitions_number=None):
    '''Function to assign to each point a given area index.

    Parameters
    ----------
    df : dask.DataFrame
        A dataframe as returned from :attr:`mobilkit.loader.load_raw_files` or imported from ``scikit-mobility`` using :attr:`mobilkit.loader.load_from_skmob`.
    tesselation_shp : str
        The path (relative or absolute) to the shapefile containing the tesselation of space. If the shapefile does not contain a `tile_ID` field it will be initialized here and included in the returned geodataframe.
    filterAreas : bool
        If tesselation is specified, keeps only the points within the specified shapofile.
    partitions_number : int, optional
        The batch size of the geopandas sjoin function to be applied. Leave it as is unless you know what you're doing.

    Returns
    -------
    df_tile : dask.dataframe
        The initial dataframe with the additional ``tile_ID`` column telling the int id of the area the point is belonging to (-1 if the point is outside of the shapefile bounds).
    tessellation_gdf : geopandas.GeoDataFrame
        The geo-dataframe with the possibly missing `tile_ID` column added.
    '''

    zones_gdf = gpd.read_file(tesselation_shp)
    if zidColName not in zones_gdf.columns:
        zones_gdf[zidColName] = np.arange(zones_gdf.shape[0], dtype=int)
    
    # This is the pipeline, it basically combines all the steps and write out the info
    def localAssign(_df):
        tmp_gdf = gpd.GeoDataFrame(_df[[latColName]],
                               geometry=gpd.points_from_xy(_df[lonColName],
                                                            _df[latColName]))
        tmp_gdf.crs = zones_gdf.crs
        df_out = gpd.sjoin(tmp_gdf[["geometry"]],
                            zones_gdf[[zidColName, "geometry"]], how="left")
        df_out = df_out[~df_out.index.duplicated()]
        df_out[zidColName].fillna(-1, inplace=True)
        _df[zidColName] = df_out[zidColName].astype(int)
        return _df

    if partitions_number is not None:
        df = df.repartition(npartitions=partitions_number)
    out_df = df.map_partitions(localAssign)
    
    if filterAreas:
        out_df = out_df[out_df[zidColName] >= 0]

    return out_df, zones_gdf


def assignAreasDF(df, zones_gdf):
    '''Returns the geo-dataframe with an additional column the `ZONE_IDX` column.
    Non overlapping areas are guaranteed to be found there with a negative -1 value;
    The order of the original index and columns is preserved.
    '''
    # Assign stop to areas
    tmp_gdf = gpd.GeoDataFrame(df[[latColName]],
                               geometry=gpd.points_from_xy(df[lonColName],
                                                            df[latColName]))
    tmp_gdf.crs = zones_gdf.crs
    df_out = gpd.sjoin(tmp_gdf[["geometry"]],
                        zones_gdf[[zidColName, "geometry"]], how="left")
    df_out = df_out[~df_out.index.duplicated()]
    df_out[zidColName].fillna(-1, inplace=True)
    df[zidColName] = df_out[zidColName].astype(int)
    return df


def plotHomeWorkPoints(uid, df_hw, gdf, ax=None, kwargs_bounds=None, kwargs_points=None):
    '''Plots the points in home and work hours for an user on the map.

    Parameters
    ----------
    uid : (str or int, depending on the uid type)
        The id of the user to plot.
    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at least the `uid`, `tile_ID` and `isHome` and `isWork` columns.
    gdf : geopandas.GeoDataFrame
        A geo-dataframe as returned by :attr:`mobilkit.spatial.tessellate`.
    ax : pyplot.axes, optional
        The axes where to plot. If ``None`` (default) creates a new figure.
    kwargs_bounds : dict, optional
        Will be passed to the geopandas plot function plotting the boundaries.
    kwargs_bounds : dict, optional
        Will be passed to the geopandas plot function plotting the boundaries.

    Returns
    -------
    ax : pyplot.axes, optional
        The axes of the figure.
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.set_aspect("equal")

    kw_bd = dict(color="None", edgecolor="k", alpha=.8, lw=2)
    if kwargs_bounds is not None:
        for k, v in kwargs_bounds.items():
            kw_bd[k] = v

    kw_pt = dict(legend=True, cmap="Set1")
    if kwargs_points is not None:
        for k, v in kwargs_points.items():
            kw_pt[k] = v

    ax = gdf.plot(ax=ax, **kw_bd)

    df_usr = df_hw[df_hw[uidColName] == uid].compute()
    points_u = gpd.GeoDataFrame(df_usr,
            geometry=gpd.points_from_xy(df_usr[lonColName],df_usr[latColName]))
    points_u["kind"] = None
    points_u.loc[points_u["isHome"]==1,"kind"] = "Home"
    points_u.loc[points_u["isWork"]==1,"kind"] = "Work"
    points_plt = points_u[~points_u["kind"].isna()].copy()

    ax = points_plt.plot("kind", ax=ax, **kw_pt)

    ax.set_xlim(points_u[lonColName].min()-.01, points_u[lonColName].max()+.01)
    ax.set_ylim(points_u[latColName].min()-.01, points_u[latColName].max()+.01)

    return ax

def plotHomeWorkUserCount(df_hw_locs, gdf, what="home", ax=None, kwargs_map=None):
    '''Plots a colormap of the number of people living (or working) in each area.

    Parameters
    ----------
    df_hw_locs : pandas.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWorkLocation` with at least the `uid`, `home_tile_ID` `work_tile_ID` columns and passed to pandas.
    gdf : geopandas.GeoDataFrame
        A geo-dataframe as returned by :attr:`mobilkit.spatial.tessellate`.
    what :  str
        The ``home`` or ``work`` string, telling whether to plot the number of people living or working in an area.
    ax : pyplot.axes, optional
        The axes where to plot. If ``None`` (default) creates a new figure.
    kwargs_map : dict, optional
        Will be passed to the geopandas plot function plotting the boundaries and colormap.

    Returns
    -------
    ax : pyplot.axes, optional
        The axes of the figure.
    gdf : geopandas.GeoDataFrame
        The original geo dataframe with an additional column (``n_users_home`` if counting home or ``n_users_work`` if counting work). If the column is already in the df it will be overwritten.
    df : pandas.DataFrame
        The ``tile_ID`` -> count of users mapping.
    '''

    assert what in ["home", "work"]

    # Count the number of people working or living in an area

    columnToUse = "home_tile_ID" if what == "home" else "work_tile_ID"
    targetColumn = "n_users_home" if what == "home" else "n_users_work"
    plotLabel = "Number of residents" if what == "home" else "Number of workers"

    locs_hh = df_hw_locs.dropna(subset=[columnToUse])\
                    .groupby(columnToUse)\
                    .agg({uidColName: 'nunique'})

    locs_hh = locs_hh.reset_index().rename(columns={uidColName: targetColumn,
                                                    columnToUse: zidColName})
    locs_hh[zidColName] = locs_hh[zidColName].astype(int)

    if targetColumn in gdf.columns:
        del gdf[targetColumn]

    gdf = pd.merge(gdf, locs_hh, on=zidColName, how="left")
    gdf[targetColumn] = gdf[targetColumn].fillna(0)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    kw_plot = dict(edgecolor="none", lw=2, cax=cax, ax=ax, legend=True)
    if kwargs_map is not None:
        for k, v in kwargs_map.items():
            kw_plot[k] = v

    ax = gdf.plot(targetColumn, **kw_plot)

    cax.set_ylabel(plotLabel, size=22)
    cax.yaxis.set_tick_params(labelsize=18)

    ax.axis("off")

    return ax, gdf, locs_hh


def plotActivityCount(df_act, gdf, what="pings", ax=None, kwargs_map=None):
    '''Plots a colormap of the number of pings (or unique users) observed in a given area in a given period.

    Parameters
    ----------
    df_act : pandas.dataframe
        A dataframe as returned by :attr:`mobilkit.spatial.areaStats` with at least the `tile_ID` and `pings` and/or `users` columns and passed to pandas.
    gdf : geopandas.GeoDataFrame
        A geo-dataframe as returned by :attr:`mobilkit.spatial.tessellate`.
    what :  str
        The ``pings`` or ``users`` string, telling whether to plot the number of pings recorded in an area or the number of unique users seen there.
    ax : pyplot.axes, optional
        The axes where to plot. If ``None`` (default) creates a new figure.
    kwargs_map : dict, optional
        Will be passed to the geopandas plot function plotting the boundaries and colormap.

    Returns
    -------
    ax : pyplot.axes, optional
        The axes of the figure.
    '''


    if what == "pings":
        plotLabel = "Pings count"
    elif what == "users":
        plotLabel = "Unique users count"
    else:
        raise RuntimeError("Unknown ``what`` '%s' in ``plotActivityCount``" % what)

    df_act = df_act[[zidColName, what]]

    if what in gdf.columns:
        del gdf[what]

    gdf = pd.merge(gdf, df_act, on="tile_ID", how="left")
    gdf[what] = gdf[what].fillna(0)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    kw_plot = dict(edgecolor="none", lw=2, cax=cax, ax=ax, legend=True)
    if kwargs_map is not None:
        for k, v in kwargs_map.items():
            kw_plot[k] = v

    ax = gdf.plot(what, **kw_plot)

    cax.set_ylabel(plotLabel, size=22)
    cax.yaxis.set_tick_params(labelsize=18)

    ax.axis("off")

    return ax


def selectAreasFromBounds(gdf, relation="within", min_lon=-99.15913, max_lon=-99.10032, min_lat=19.41353, max_lat=19.46100,):
    '''Function to select areas from a geodataframe given the bounds of a selected region.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geodataframe with at least the ``tile_ID`` and ``geometry`` columns as returned by :attr:`mobilkit.spatial.tessellate`.
    relation : str, optional
        The relation between the bounds and the areas. "within" or "intersects"
    min/max_lon/lat : float, optional
        The minimum and maximum latitude and longitude of the box.

    Returns
    -------
    areas_ids : set
        The set of the areas **within** or **intersecting** the given bounds
    '''

    assert relation in ["within", "intersects"]

    selected_polygon = Polygon([
            [min_lon, min_lat],
            [min_lon, max_lat],
            [max_lon, max_lat],
            [max_lon, min_lat],
            [min_lon, min_lat]
        ])

    if relation == "within":
        idxs = gdf.within(selected_polygon)
    elif relation == "intersects":
        idxs = gdf.intersects(selected_polygon)
    return set([int(i) for i in gdf[idxs][zidColName].values])


def replaceAreaID(df, mapping):
    '''Function that replaces all the ``tile_ID`` with a new id given in the mapping.

    Parameters
    ----------
    df : dask.DataFrame
        A dataframe with at least the ``tile_ID`` column.
    mapping : dict
        A mapping between the original ``tile_ID`` and the new desired one. MUST CONTAIN ALL THE ``tile_ID`` s present in df.

    Returns
    -------
    df_out : dask.DataFrame
        A copy of the original dataframe with the ``tile_ID`` replaced.

    '''
    df = df.map_partitions(lambda p: p.replace({zidColName: mapping}))
    return df


def meanshift(df, bw=0.01, maxpoints=100, **kwargs):
    '''
    Given the points of a user finds the home location with MeanShift
    clustering.
    
    Parameters
    ----------
    df : pandas.DataFrame
        With at least `latcol,loncol`.
    bw : float
        Bandwidth to be used in MeanShift.
    maxpoints : int
        The maximum number of points to be used in meanshift.
        If more, a fraction of the df to have `maxpoints` will be sampled.
    kwargs
        Will be passed to `sklearn.cluster.MeanShift` constructor.
        
    Returns
    -------
    clust_center : tuple
        The center of the cluster found in the `(longitude,latitude)` format.
     
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like *Cuebiq*, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''
    if len(df)>maxpoints:
        df = df.sample(frac=maxpoints/df.shape[0])
    ms = cluster.MeanShift(bandwidth=bw, **kwargs)
    ms.fit(df[[lonColName,latColName]])
    labels = ms.labels_
    counts = np.bincount(labels)
    most = np.argmax(counts)
    cluster_centers = ms.cluster_centers_
    return cluster_centers[most]


def haversine_pairwise(X,Y=None):
    '''
    Parameters
    ----------
    X, Y : np.array
        a Nx * 2 and Ny*2 arrays of (lat,lon) coordinates.
    
    Returns
    -------
    distances : np.array
        a Nx*Ny matrix of distances in kilometers
    '''
    X = np.radians(X)
    if Y is None:
        Y = X
    else:
        Y = np.radians(Y)
        
    distances = haversine_distances(X, Y)
    distances *= 6371000/1000 # multiply by Earth radius to get kilometers
    
    return distances


def rad_of_gyr(coords):
    '''
    Parameters
    ----------
    coords : np.array
        a Nx*2 array of (lat,lon) coordinates.
    
    Returns
    -------
    radius_of_gyrations : float
        The radius of gyration for the selected coords.
    '''
    
    com = np.mean(coords, axis=0, keepdims=True)
    rog_sum = (haversine_pairwise(coords, com)**2.).sum()
    rog = np.sqrt(rog_sum / max(1,coords.shape[0]))
    
    return rog
    

    
def total_distance_traveled(coords):
    '''
    Parameters
    ----------
    coords : np.array
        a Nx*2 array of (lat,lon) coordinates.
    
    Returns
    -------
    total_distance_traveled : float
        The radius of gyration for the selected coords.
    '''
    tot_dist = 0
    for i, j in zip(coords[:-1,:], coords[1:,:]):
        tot_dist += haversine_pairwise([i],[j])[0][0]
    
    return tot_dist
    
# POIs tools

def convert_df_crs(df, lat_col="lat", lon_col="lng",
                   from_crs="EPSG:4326", to_crs="EPSG:6362"):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the `lat_col` and `lon_col` columns at least.
    lat_col, lon_col : str
        The names of the columns containing the latitude and longitude of the
        points in `df`.
    from_crs, to_crs : str
        The codes of the original and target projections to use.
        
    Returns
    -------
    df : pd.DataFrame
        The original data frame with two additional columns named `lat_col + '_proj'`
        and `lon_col + '_proj'` containing the original coordinates projected to
        `to_crs`.
    '''
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col],df[lat_col]))
    gdf.set_crs(from_crs, inplace=True)
    gdf.to_crs(to_crs, inplace=True)
    df = df.assign(**{
            lon_col + "_proj": gdf["geometry"].x.values,
            lat_col + "_proj": gdf["geometry"].y.values})
    return df

def distanceHomeUser(g,
                     lon_col=lonColName, lat_col=latColName,
                     h_lon_col="homelon", h_lat_col="homelat"):
    '''
    Parameters
    ----------
    g : pandas.DataFrame
        A dataframe containing at least the `lat_col` and `lon_col` columns with
        the raw points' coordinates and the home coordinates in `homelon` and
        `homelat` columns. **Must contain all the data of one user only.**
    lat_col, lon_col : str
        The names of the columns containing the latitude and longitude of the
        points in `g`.
    h_lat_col, h_lon_col : str
        The names of the columns containing the latitude and longitude of the
        home user `g`.
        
    Returns
    -------
    g : pd.DataFrame
        The original data frame with an additional column named `'home_dist'`
        containing the haversine distance between each point and the home location
        **in kilometers**.
        
    
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like *Cuebiq*, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''
    home_lat_lon = g[[h_lat_col,h_lon_col]].iloc[:1].values
    tmp_coordinates = g[[lat_col,lon_col]].values
    distances = haversine_pairwise(tmp_coordinates, home_lat_lon)
    g["home_dist"] = distances[:,0]
    return g

def distanceHomeDF(g, **kwargs):
    '''
    Parameters
    ----------
    g : pandas.DataFrame
        A dataframe containing at least the `lat_col` and `lon_col` columns with
        the raw points' coordinates and the home coordinates in `homelon` and
        `homelat` columns of all the users.
    **kwargs
        Such as `lat_col`, `lon_col` will be passed to :attr:`mobilkit.spatial.distanceHomeUser`.
        
    Returns
    -------
    g : pd.DataFrame
        The original data frame with an additional column named `'home_dist'`
        containing the haversine distance between each point and the home location
        of the user of that row **in kilometers**.
    '''
    if g.shape[0] > 0:
        g = g.groupby(uidColName).apply(distanceHomeUser, **kwargs)
    else:
        g["home_dist"] = []
    return g

def compute_poi_index_dist(g, tree_model=None,
                           lon_col="lng_proj", lat_col="lat_proj"):
    '''
    Parameters
    ----------
    g : pandas.DataFrame
        A dataframe containing at least the `lat_col` and `lon_col` columns with
        the raw points' coordinates projected to the same projection of the `tree_model`.
    tree_model : sklearn.neighbors.KDTree
        A KDTree trained on the POIs projected in the same proj of lat on lon points.
        Distance will be computed by the tree.
    lat_col, lon_col : str
        The names of the columns containing the latitude and longitude of the points in `g`.
        By default they match the ones used in :attr:`mobilkit.spatial.compute_poi_visit`.
       
    Returns
    -------
    g : pd.DataFrame
        The original data frame with two additional columns named:
            - `'poi_distance'` the distance of the closest poi found in the tree
                **in KM**;
            - `'_POI_INDEX_'` the 0-based index of the closest tree leaf.
    '''
    if g.shape[0] > 0:
        distances, poi_indexes = tree_model.query(g[[lon_col,lat_col]].values)
        distances = distances[:,0] / 1000. # In Km
        poi_indexes = poi_indexes[:,0]
    else:
        distances = []
        poi_indexes = []
    g["poi_distance"] = distances
    g["_POI_INDEX_"] = poi_indexes
    return g

def filter_to_box(df, minlon, maxlon, minlat, maxlat,
                 lat_col=latColName, lon_col=lonColName):
    '''
    Parameters
    ----------
    df : DataFrame
        A dataframe containing at least the `lat_col` and `lon_col` columns with
        the raw points' coordinates.
    {min,max}{lat,lon} : float
        The min and max values of lat and lon (will keep all coords >= min and <= max).
    lat_col, lon_col : str
        The names of the columns containing the latitude and longitude of the points in `df`.
       
    Returns
    -------
    df : pd.DataFrame
        The original data frame filtered to the points within the box.
    '''
    return df[(df[lat_col].between(minlat, maxlat))
              & (df[lon_col].between(minlon, maxlon))]
    
def compute_poi_visit(df_pings, df_homes, df_POIs,
                      from_crs="EPSG:4326", to_crs="EPSG:6362",
                      min_home_dist_km=.2, visit_time_bin="1H",
                      lat_lon_tol_box=.02):
    '''
    Computes the set of users and number of users visiting a given POI for each
    `visit_time_bin` period of time found in the pings dataframe.
    
    Parameters
    ----------
    df_ping : dask.DataFrame
        A dataframe containing at least the `uid`, `datetime`, `lat` and `lng`
        columns with the raw points' coordinates.
        The coordinates must be given in the `from_crs` projection.
    df_homes : Dataframe
        A `pandas` or `dask` Dataframe with the `uid`, `homelat` and `homelon`
        home coordinates of all the users in the df.
        The coordinates must be given in the `from_crs` projection.
        Note that the three dataframes of pings, homes and POIs **must** feature
        the same initial projection equal to `from_crs`.
    df_POIs : Dataframe
        A `pandas` or `dask` Dataframe with at least the `radius`, `poilat` and
        `poilon` columns stating the radius to be considered in the POI (in km)
        and the POI's coordinates.
        The coordinates must be given in the `from_crs` projection.
    from_crs, to_crs : str
        The codes of the original and target projections to use. Will be used to
        compute planar distances in km using a euclidean distance so use the
        appropriate reference system for your ROI (e.g., use `to_crs='EPSG:6362''`
        for the Mexico area).
        Will be passed to :attr:`mobilkit.spatial.convert_df_crs`.
    min_home_dist : float
        The minimum distance for a point to be from the user's home to be considered
        valid (in km).
    visit_time_bin : str
        The frequency of the time bin to use. Each `datetime` will be floored to this
        time frequency.
    lat_lon_tol_box : float
        The pings will be filtered within the box of the maximum/minimum
        latitude/longitude of the POIs original projection's dataframe.
        This is the margin added around this box to account for pings right outside
        of the POIs' boundaries that may still fall into their radius.
       
    Returns
    -------
    pings_merged_home_poi, results : dask.DataFrame, pd.DataFrame
        - `pings_merged_home_poi` is a view on the `dask.DataFrame` containing, for all
            the points falling within the POIs radius and far enough from users' home:
                - the original pings columns plus their projected coords in `{lat,lng}_proj`;
                - the `home` and 'poi' original and projected (with '_proj' suffix) lat coords;
                - 'poi_distance', '_POI_INDEX_' the distance (in km) and the unique index of the
                    closest POI;
                - all the `df_POIs` columns related to this POI (if common names of columns are
                    found they will be inserted with the `_FROM_POI_TABLE` suffix);
                - 'home_dist' the distance in km from the user's home;
                - 'time_bin' the original datetime floored to `visit_time_bin` freq.
        - `results` is a dataframe containing, for each unique `_POI_INDEX_` and `time_bin` as
            given by `visit_time_bin`:
                - all the `df_POIs` columns related to this POI;
                - `users,num_users` the columns containing the list of the `uid`-s of the users
                    found in that POI and that `time_bin` and their number.
    '''
    # prepare aux DFs
    if type(df_homes) == pd.DataFrame:
        df_homes = dd.from_pandas(df_homes, npartitions=1)
    df_homes = df_homes.repartition(npartitions=1)
    if type(df_POIs) == pd.DataFrame:
        df_POIs = dd.from_pandas(df_POIs, npartitions=1)
    df_POIs = df_POIs.repartition(npartitions=1)
    df_POIs = df_POIs.assign(_POI_INDEX_=da.arange(df_POIs.shape[0].compute(), dtype=np.int64))
    
    # Limiting to box
    df_POIs_pd = df_POIs.compute()
    maxlat, minlat = df_POIs_pd["poilat"].max()+lat_lon_tol_box,\
                        df_POIs_pd["poilat"].min()-lat_lon_tol_box
    maxlon, minlon = df_POIs_pd["poilon"].max()+lat_lon_tol_box,\
                        df_POIs_pd["poilon"].min()-lat_lon_tol_box
    
    # Project the home locations, raw pings and POIs locations to the selected crs
    projected_pings = filter_to_box(df_pings, minlon, maxlon, minlat, maxlat)
    projected_pings = projected_pings.map_partitions(convert_df_crs, from_crs=from_crs, to_crs=to_crs)
    projected_POIs = df_POIs.map_partitions(convert_df_crs, from_crs=from_crs, to_crs=to_crs,
                                           lat_col="poilat", lon_col="poilon",)
    projected_homes = df_homes.map_partitions(convert_df_crs, from_crs=from_crs, to_crs=to_crs,
                                            lat_col="homelat", lon_col="homelon",)
    
    # Prepare the Kdtree model
    tree_model = KDTree(projected_POIs[["poilon_proj","poilat_proj"]].compute(), leaf_size=20, metric="euclidean")

    # Check all is good
    assert 1 == projected_POIs._POI_INDEX_.value_counts().compute().max()
    # Check that we do not have double POIs
    distances, poi_indexes = tree_model.query(projected_POIs[["poilon_proj","poilat_proj"]].values)
    assert np.all(poi_indexes[:,0] == np.arange(len(poi_indexes), dtype=int))
    
    # Add the home location
    pings_merged_home = projected_pings.merge(projected_homes, on="uid", how="inner")
    
    pings_merged_home_poiID = pings_merged_home.map_partitions(compute_poi_index_dist,
                                                               tree_model=tree_model)
    pings_merged_home_poi = pings_merged_home_poiID.merge(projected_POIs,
                                                          on="_POI_INDEX_",
                                                          how="inner",
                                                          suffixes=("","_FROM_POI_TABLE"))
    
    # Filter on poi dist radius, then home on remaining
    pings_merged_home_poi = pings_merged_home_poi[pings_merged_home_poi["poi_distance"]
                                                  <= pings_merged_home_poi["radius"]]
    pings_merged_home_poi = pings_merged_home_poi.map_partitions(distanceHomeDF)
    pings_merged_home_poi = pings_merged_home_poi[pings_merged_home_poi["home_dist"] > min_home_dist_km]
    
    # Round datetime to time bin
    pings_merged_home_poi = pings_merged_home_poi.assign(time_bin=
                                                         pings_merged_home_poi["datetime"]\
                                                                 .dt.floor(visit_time_bin))
    
    # Count the users per poi and time bin
    def tmp_foo(g):
        usrs = list(g["uid"].unique())
        n_us = len(usrs)    
        return pd.DataFrame([[usrs, n_us]], columns=["users", "num_users"])

    results = pings_merged_home_poi.groupby(["_POI_INDEX_","time_bin"]).apply(tmp_foo,
                                                                              meta={
                                                                                  "users": object,
                                                                                  "num_users": int,
                                                                              }).compute()
    # merge with df_POIs_pd
    results = results.join(df_POIs_pd.set_index("_POI_INDEX_"), how="inner").reset_index()
    
    return pings_merged_home_poi, results
    

def compute_population_density(df, **kwargs):
    '''
    Parameters
    ----------
    df : dask.DataFrame
        A dataframe as returned by :attr:`mobilkit.temporal.filter_daynight_time` with
        at least the `date,daytime,uid,lat,lng` columns containing the date rounded to
        day, a bool stating if the point is in daytime or nightime, the user id and the
        coordinates of the point.
    **kwargs
        Will be passed to :attr:`mobilkit.spatial.meanshift`.
    
    Returns
    -------
    df : pandas.DataFrame
        A dataframe with a multi index of `date,daytime,uid` and as columns the `lat`
        and `lng` coordinates of the mean shift location of the user on that part of
        the day on that date.
    '''
    res = df.groupby(["date","daytime","uid"]).apply(meanshift, **kwargs).compute()
    res = pd.DataFrame(res, columns=["pppointtt"])
    res = res.assign(**{latColName: res['pppointtt'].apply(lambda v: v[1]),
                         lonColName: res['pppointtt'].apply(lambda v: v[0])})
    res = res[[latColName, lonColName]].copy()
    return res


def density_map(latitudes, longitudes, center, bins, radius):
    '''
    Parameters
    ----------
    latitudes, longitudes : array-like
        The arrays containing the latitude and longitude coordinates of each user's
        location.
    center : tuple-like
        The (lat, lon) of the center where to compute the population density around.
    bins : int
        The number of bins to use horizontally and vertically in the region around the
        center.
    radius : float
        The space to consider above, below, left and right of the center (same unity
        of the center).
    
    Returns
    -------
    density : np.array
        The 2d histogram of the population.
    
    '''
    # Center the map around the provided center coordinates
    histogram_range = [
        [center[1] - radius, center[1] + radius],
        [center[0] - radius, center[0] + radius]
    ]
    res = np.histogram2d(longitudes, latitudes, bins=bins, range=histogram_range)
    return res

from pandas import IndexSlice

def stack_density_map(df, dates, center, daytime=True, bins=200, radius=1):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe as returned by :attr:`mobilkit.spatial.compute_population_density`
        with a multi index of `date,daytime,uid` and as columns the `lat` and `lng`
        coordinates of the mean shift location of the user on that part of
        the day on that date.
    dates : list pof datetime
        A list of dates when to compute the density.
    center : tuple-like
        The (lat, lon) of the center where to compute the population density around.
    daytime : bool
        Whether to compute the density on the daytime or nightime part of selected
        dates.
    bins : int
        The number of bins to use horizontally and vertically in the region around the
        center.
    radius : float
        The space to consider above, below, left and right of the center (same unity
        of the center).
    
    Returns
    -------
    maps, results : np.array, tuple
        `maps` is the tensor of shape `(len(dates),bins,bins)` storing for each date the x-y
        density map as computed by :attr:`mobilkit.spatial.density_map`.
        `results` stores the x and y bins.
    '''
    maps = np.zeros((len(dates),bins,bins))
    slicer = IndexSlice
    for iii, date in enumerate(dates):
        tmp_rows = df.loc[slicer[date,daytime,:],:]
        tmp_res = density_map(tmp_rows[latColName], tmp_rows[lonColName],
                              center=center, bins=bins, radius=radius)
        maps[iii,:,:] = tmp_res[0].copy()
    return maps, tmp_res[1:]

import numpy.ma as ma
def stats_density_map(df, dates, center, daytime=True, bins=200, radius=1, clip_std=1e-4):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe as returned by :attr:`mobilkit.spatial.compute_population_density`
        with a multi index of `date,daytime,uid` and as columns the `lat` and `lng`
        coordinates of the mean shift location of the user on that part of
        the day on that date.
    dates : list pof datetime
        A list of dates when to compute the density.
    center : tuple-like
        The (lat, lon) of the center where to compute the population density around.
    daytime : bool
        Whether to compute the density on the daytime or nightime part of selected
        dates.
    bins : int
        The number of bins to use horizontally and vertically in the region around the
        center.
    radius : float
        The space to consider above, below, left and right of the center (same unity
        of the center).
    clip_std : float
        Pixels with a 0 or `nan` std will be clipped to this value when computing the
        z-score. The same pixels will be set to -1 on output.
    
    Returns
    -------
    results : dict
        A dictionary containing teh key-values:
            - `stack` the tensor of shape `(len(dates),bins,bins)` storing for each date
                the x-y density map as computed by :attr:`mobilkit.spatial.density_map`.
            - `avg`, `std` the average and standard deviation population density with
                shape `(1,bins,bins)`.
            - `x_bins`, `y_bins` the bins of the 2d histogram as produced by
                :attr:`mobilkit.spatial.density_map`.
    '''
    tensor_maps, bins = stack_density_map(df, dates, center, daytime=daytime, bins=bins, radius=radius)
    maps_avg = tensor_maps.mean(axis=0, keepdims=True)
    maps_std = tensor_maps.std(axis=0, keepdims=True)
    mask = np.where(np.logical_or(np.isnan(maps_std), maps_std==.0))
    maps_std = np.where(np.logical_or(np.isnan(maps_std), maps_std==.0),
                        np.ones_like(maps_std)*clip_std, maps_std)
    maps_zsc = np.zeros_like(tensor_maps)
    maps_zsc = (tensor_maps - maps_avg) / maps_std
    
    maps_std[mask] = -1

    return {"stack": tensor_maps, "zsc": maps_zsc,
            "avg": maps_avg, "std": maps_std,
            "x_bins": bins[0], "y_bins": bins[1]}


def box2poly(box):
    '''
    [min_lon, min_lat, max_lon, max_lat]
    '''
    return shapely.geometry.Polygon([[box[0],box[3]], [box[2],box[3]], [box[2],box[1]],
                                [box[0],box[1]], [box[0],box[3]]])

def makeVoronoi(gdf):
    x = gdf["geometry"].x.values
    y = gdf["geometry"].y.values

    coords = np.vstack((x, y)).T
    vor = Voronoi(coords)

    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in 
        vor.ridge_vertices if -1 not in line]
    polys = shapely.ops.polygonize(lines)
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=gdf.crs)

    tmp_box = box2poly(gdf.unary_union.bounds)
    tmp_gdf = gpd.GeoDataFrame(["box"], geometry=[tmp_box], crs=gdf.crs)
    voronois = gpd.overlay(voronois, tmp_gdf)
    
    return voronois