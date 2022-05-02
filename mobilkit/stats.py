# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

'''Tools and functions to compute the per-users and per area stats.

.. note::
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
'''
import dask
from dask import dataframe as dd
from dask import array as da
import dask.bag as db

from mobilkit.dask_schemas import nunique
from mobilkit.spatial import convert_df_crs, userHomeWorkDistance
from mobilkit.tools import userHomeWorkTravelTimeOSRM

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from copy import copy
from collections import Counter

from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)

def userStats(df):
    '''Computes the stats per user:
        - days spanned (time from first to last ping);
        - days active (actual number of days being active);
        - number of pings per user;
        - number of pings per user per active day;

    Parameters
    ----------
    df : dask.dataframe
        The dataframe extracted or imported. Must contains the ``uid`` column and the ``datetime`` one.

    Returns
    -------
    df_out : dask.DataFrame
        A dask dataframe with the stats per user in three columns:
            - ``daysActive`` the number of different days where the user has been active;
            - ``daysSpanned`` the days spanned between the first and last recorded ping;
            - ``pings`` the number of pings recorded for the user;
            - ``pingsPerDay`` the number of pings recorded for the user in every active day;
            - ``avg`` the average number of pings per recorded day for the user.

    Example
    -------

    >>> df_out = mk.stats.userStats(df)
    >>> df_out.head()
    uid   |    min_day |    max_day |  pings |  daysActive |   avg | daysSpanned | pingsPerDay    |
    'abc' | 2017-08-12 | 2017-12-22 |   3452 |         124 | 27.83 |         222 | [12,22,...,13] |

    '''

    df = df.assign(day=df[dttColName].dt.floor("1d"))
    meta_original = dict(**df.dtypes)
    return_meta = {k: meta_original[k] for k in [uidColName]}
    return_meta["min_day"] = meta_original["day"]
    return_meta["max_day"] = meta_original["day"]
    return_meta["pings"] = np.int64
    return_meta["daysActive"] = np.int64
    return_meta["daysSpanned"] = np.int64
    return_meta["pingsPerDay"] = object
    return_meta["avg"] = np.float64
    
    out = df.assign(agg_key=df[uidColName])\
                .groupby("agg_key")\
                .apply(_computeUserStats, meta=return_meta)

    return out.reset_index(drop=True)[[uidColName, "min_day", "max_day", "pings",
                      "daysActive", "daysSpanned", "pingsPerDay", "avg"]]

def _computeUserStats(df):
    '''
    Help function to compute user stats inplace.
    '''
    day_pings_counts = Counter(df["day"].values)
    min_day, max_day = df["day"].min(), df["day"].max()
    spanned = max(1, (max_day - min_day).days)
    daysActive = len(day_pings_counts)
    pings_per_day = [v for k, v in sorted(day_pings_counts.items())]
    tot_pings = df.shape[0]
    avg_pings = tot_pings / daysActive
    tmp_uid = df[uidColName].iloc[0]
    
    return pd.DataFrame([[tmp_uid, min_day, max_day, tot_pings,
                          daysActive, spanned, pings_per_day, avg_pings]],
                        columns=[uidColName, "min_day", "max_day", "pings",
                          "daysActive", "daysSpanned", "pingsPerDay", "avg"])
    

def filterUsersFromSet(df, users_set):
    '''
    Function to filter the pings and keep only the ones of the users in ``users_set``.

    Parameters
    ----------
    df : dask.dataframe
        The dataframe containing the pings.
    users_set : set or list
        The ids of the users to keep.

    Returns
    -------
    df_out : dask.dataframe
        The filtered dataframe containing the pings of the valid users only.
    '''
    df_out = df[df[uidColName].isin(users_set)]

    return df_out

def filterUsers(df, dfStats=None, minPings=1, minDaysSpanned=1, minDaysActive=1, minSuperUserDayFrac=None, superUserPingThreshold=None):
    '''
    Function to filter the pings and keep only the ones of the users with given statistics.

    Parameters
    ----------
    df : dask.dataframe
        The dataframe containing the pings.
    dfStats : dask.dataframe, optional
        The dataframe containing the pre-computed stats of the users as returned by :attr:`mobilkit.stats.userStats`. If ``None``, it will be automatically computed. In either cases it is returned together with the result.
    minPings : int
        The minimum number of recorded pings for a user to be kept.
    minDaysSpanned : float
        The minimum number of days between the first and last ping for a user to be kept.
    minDaysActive : int
        The minimum number of active days for a user to be kept.
    minSuperUserDayFrac : float
        The minimum fraction of days with same or more pings than ``superUserPingThreshold`` for a user to be considered. Must be between 0 and 1.
    superUserPingThreshold : int
        The minimum number of pings for a user-day to be considered as super user.

    Returns
    -------
    df_out, df_stats, valid_users_set : dask.dataframe, dask.dataframe, set
        The dataframe containing the pings of the valid users only, the one containing the stats per user and the set of the valid users.

    '''
    if minSuperUserDayFrac is not None:
        assert superUserPingThreshold is not None
        superUserPingThreshold = int(superUserPingThreshold)
        assert 0 < minSuperUserDayFrac <= 1

    if dfStats is None:
        dfStats = userStats(df)

    valid_users = dfStats[
            (dfStats["pings"] >= minPings)
            & (dfStats["daysActive"] >= minDaysActive)
            & (dfStats["daysSpanned"] >= minDaysSpanned)
        ]
    
    if minSuperUserDayFrac is not None:
        valid_users["superDays"] = valid_users["pingsPerDay"].map(
                                        lambda l: sum([d>superUserPingThreshold for d in l]),
                                        meta=('superDays','int'))
        valid_users["fracSuperDays"] = valid_users["superDays"] / valid_users["daysActive"]
        
        valid_users = valid_users[valid_users["fracSuperDays"] >= minSuperUserDayFrac]
        
    valid_users_set = set(valid_users[uidColName].unique().compute())
    
    df_out = df[df[uidColName].isin(valid_users_set)]

    return df_out, dfStats, valid_users_set


def userHomeWork(df, homeHours=(19.5,7.5), workHours=(9.,18.5), weHome=False):
    '''Computes, for each row of the dataset, if the ping has been recorded in home or
    work time. Can be used in combination with :attr:`mobilkit.stats.determineHomeWork` to determine the home and work location of a user.

    Parameters
    ----------
    df : dask.dataframe
        The loaded dataframe with at least `uid`, `datetime` and `tile_ID` columns.
t 
    homeHours :  tuple, optional
        The starting and end hours of the home period in 24h floating numbers. For example, to put the house period from 08:15pm to 07:20am put ``homeHours=(20.25, 7.33)``.
    workHours :  tuple, optional
        The starting and end hours of the work period in 24h floating numbers. For example, to put the work period from 09:15am to 06:50pm put ``workHours=(9.25, 18.8333)``.
        **Note that work hours are counted only from Monday to Friday.**
    weHome : bool, optional
        If ``False`` (default) counts only weekend hours within the home hours as valid home hours.
        If ``True``, all the pings recorded during the weekend (Saturday and Sunday) are counted as home pings.

    Returns
    -------
    out : dask.dataframe
        The dataframe with two additional columns: `isHome` and `isWork` telling if a given ping has been recorded during home or work time (or none of them).
    
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''

    # Note that day of week in spark is from 1 (Sunday) to 7 (Saturday).
    cols_in = list(df.columns)
    
    # Lazy copy to start new stack of computations
    df = copy(df)
    df = df.assign(dow=df[dttColName].dt.weekday,
                    hfloat=df[dttColName].dt.hour + df[dttColName].dt.minute / 60.)

    
    if weHome:
        df["isHome"] = df["dow"] > 4
    else:
        df["isHome"] = False
        
    if homeHours[0] < homeHours[1]:
        df["isHome"] = (df["isHome"] | (df["hfloat"].between(*homeHours)))
    else:
        df["isHome"] = (df["isHome"] | (
                                (df["hfloat"] >= homeHours[0])
                                | (df["hfloat"] < homeHours[1])
                            ))

    df["isWork"] = df["dow"] < 5
    if workHours[0] < workHours[1]:
        df["isWork"] = df["isWork"] & df["hfloat"].between(*workHours)
    else:
        df["isWork"] = df["isWork"] & (
                            (df["hfloat"] >= workHours[0])
                            | (df["hfloat"] < workHours[1])
                        )
    cols_out = cols_in + ["isHome", "isWork"]
    out = df[cols_out]
    
    return out

def homeWorkStats(df_hw):
    '''Given a dataframe returned by :attr:`mobilkit.stats.userHomeWork` computes, for each
    user and area, the total number of pings recorded in that area (``total_pings`` column),
    the pings recorded in home hours (``home_pings`` column) and the ones in work hours
    (``work_pings`` column).

    Parameters
    ----------

    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at least the `uid`,
        `tile_ID` and `isHome` and `isWork` columns.

    Returns
    -------
    df_hw_stats : dask.dataframe
        The dataframe containing, for each user and area id:
            - ``total_pings``: the total number of pings recorded for that user in that area
            - ``home_pings``: the pings recorded for that user in home hours in that area
            - ``work_pings``: the ping in work hours for that user in that area

    Note
    ----
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''
    meta_original = dict(**df_hw.dtypes)
    return_meta = {k: meta_original[k] for k in [uidColName, zidColName]}
    return_meta["total_pings"] = np.int64
    return_meta["home_pings"] = np.int64
    return_meta["work_pings"] = np.int64
    
    df_out = df_hw.groupby([uidColName, zidColName]).apply(_computeUserHW, meta=return_meta)

    return df_out

def _computeUserHW(df):
    '''
    Help function to compute user home work stats inplace.
    '''
    tot_pings = df.shape[0]
    home_pings = df["isHome"].sum()
    work_pings = df["isWork"].sum()

    tmp_uid = df[uidColName].iloc[0]
    tmp_zid = df[zidColName].iloc[0]
    
    return pd.DataFrame([[tmp_uid, tmp_zid, tot_pings, home_pings, work_pings]],
                        columns=[uidColName, zidColName,
                            "total_pings", "home_pings", "work_pings"])
    

def areaStats(df, start_date=None, stop_date=None, hours=(0,24), weekdays=(1,2,3,4,5,6,7)):
    '''Computes the stats of a given area in terms of pings and unique users seen in a given
    area in a given period.

    Parameters
    ----------
    df : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.spatial.tessellate` with at least the `uid`,
        `tile_ID` and `datetime` columns.
    start_date : datetime.datetime
        A python datetime object with no timezone telling the date (included) to start from.
        The default behavior is to keep all the events.
    stop_date : datetime.datetime, optional
        A python datetime object with no timezone telling the date (excluded) to stop at.
        Default is to keep all the events.
    hours : tuple, optional
        The hours when to start (included) and stop (excluded) in float notation
        (e.g., 09:15 am is 9.25 whereas 10:45pm is 22.75).
    weekdays : tuple or set or list, optional
        The list or tuple or set of days to be kept in python notation so 0 = Monday,
        1 = Tuesday, ... and 6 = Sunday.

    Returns
    -------
    df : dask.DataFrame
        The ``tile_ID`` -> count of pings/users mapping.
    '''

    df_act = df[[uidColName,zidColName,dttColName]]
    if start_date is not None:
        df_act = df_act[df_act[dttColName] >= start_date]
    if stop_date is not None:
        df_act = df_act[df_act[dttColName] < stop_date]

    df_act["hour"] = df_act[dttColName].dt.hour + df_act[dttColName].dt.minute / 60
    df_act["dow"] = df_act[dttColName].dt.weekday
    
    df_act = df_act[
            (df_act["dow"].isin(set(weekdays)))
            &  (df_act["hour"].between(*hours))
        ]
        
    df_act = df_act.groupby(zidColName).agg({uidColName: nunique, "dow": "count"}).reset_index()
    df_act = df_act.rename(columns={uidColName: "users", "dow": "pings"})
                         
    return df_act


def userHomeWorkLocation(df_hw : dask.dataframe, force_different: bool=False):
    '''Given a dataframe returned by :attr:`mobilkit.stats.userHomeWork` computes, for each user, the home and work area as well as their location.
    The home/work area is the one with more pings recorded and the location is assigned to the mean point of this cloud.

    Parameters
    ----------

    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at least the `uid`, `tile_ID` and `isHome` and `isWork` columns.
    force_different :  bool, optional
        Whether we want to force the work location to be different from the home location.

    Returns
    -------
    df_hw_locs : dask.dataframe
        A dataframe containing, for each ``uid`` with at least one ping at home or work:
            - ``pings_home``: the total number of pings recorded in the home area
            - ``pings_work``: the total number of pings recorded in the work area
            - ``tile_ID_home``: the tile id of the home area
            - ``tile_ID_work``: the tile id of the work area
            - ``lng_home``: the longitude of the home location
            - ``lat_home``: the latitude of the home location
            - ``lng_work``: the longitude of the work location
            - ``lat_work``: the latitude of the work location
            
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''
    # Same as before, focus on the points on the zone and times and compute the average point.
#     w_u = pyspark.sql.Window().partitionBy(uidColName)
    meta = {
        'tot_pings': np.int64,
        'home_'+zidColName: np.int64,
        latColName+'_home': np.float64,
        lonColName+'_home': np.float64,
        'home_pings': np.int64,
        'work_'+zidColName: np.int64,
        latColName+'_work': np.float64,
        lonColName+'_work': np.float64,
        'work_pings': np.int64,
    }
    df_hw_loc = df_hw.groupby([uidColName])\
                    .apply(
                        _determine_home_work_user,
                        force_different=force_different,
                        meta=meta
                    )

    return df_hw_loc


def _determine_home_work_user(df, force_different: bool=False):
    '''
    Help function to compute at once the home and work location of users.
    Parameters
    ----------
    df : dask.dataframe
        The slice of the df returned by :attr:`mobilkit.stats.userHomeWork` when groupbed by user id.
    force_different : bool, optional
        Whether or not to force the work tile to be different from the home one.
    '''
    n_pings = df.shape[0]
    uid = df[uidColName].iloc[0]
    cnt_hw = df[df['isHome'] | df['isWork']].groupby(zidColName)[["isHome","isWork"]].agg("sum")
    if cnt_hw.shape[0] == 0:
        home_tile = work_tile = None
        home_pings = work_pings = {
            latColName: None,
            lonColName: None,
            'n_pings': 0
        }
    else:
        home_tile = cnt_hw.reset_index().sort_values('isHome', ascending=False).iloc[0][zidColName]
        home_pings = df[(df['isHome'] == True) & (df[zidColName] == home_tile)].agg({
            latColName: 'mean',
            lonColName: 'mean',
            zidColName: 'count',
        }).rename({zidColName: 'n_pings'})
        
        if force_different:
            cnt_hw = cnt_hw[cnt_hw.index != home_tile]
        if cnt_hw.shape[0] > 0:
            work_tile = cnt_hw.reset_index().sort_values('isWork', ascending=False).iloc[0][zidColName]
            work_pings = df[(df['isWork'] == True) & (df[zidColName] == work_tile)].agg({
                latColName: 'mean',
                lonColName: 'mean',
                zidColName: 'count',
            }).rename({zidColName: 'n_pings'})
        else:
            work_tile = None
            work_pings = {latColName: None, lonColName: None, 'n_pings': 0}
    
    return pd.Series({'tot_pings': n_pings,
                      'home_'+zidColName: home_tile,
                      latColName+'_home': home_pings[latColName],
                      lonColName+'_home': home_pings[lonColName],
                      'home_pings': home_pings['n_pings'],
                      'work_'+zidColName: work_tile,
                      latColName+'_work': work_pings[latColName],
                      lonColName+'_work': work_pings[lonColName],
                      'work_pings': work_pings['n_pings'],
                     })


# Functions for spatial statistics
def userBasedBufferedStat(df_stat, df_user_grid, stat_col,
                          uid_col=uidColName,
                          tile_col=zidColName,
                          explode_col=False,
                          how='inner',
                          stats=['min','max','mean','std','count'],
                         ):
    '''
    Given a dataframe containing the per user stat `df_stat` in the `stat_col`
    and a dataframe containing the users per area as returned from `mk.stats.computeBufferStat`
    computes the `stats` of the `stat_col` merging the two df on the `tile_col`.
    
    Parameters
    ----------
    df_stat : pd.DataFrame
        The dataframe containing at least the `uid_col` and `stat_col`. They can also be in the
        df's index as it will be reset.
    df_user_grid : pd.DataFrame
        A dataframe containing the users per area (in the `uid_col` and `tile_col`)as returned
        from `mk.stats.computeBufferStat` using passing as `gdf_stat` the home work locations,
        `lat_name='lat_home',lon_name='lng_home'`, `column=uidColName` and `aggregation=set`. 
    uid_col, tile_col : str, optional
        The columns containing the user id and the tile id in the two input dfs.
    explode_col : bool, optional
        Whether we need to explode the `stat_col` before merging (for list-like observations).
    how : str, optional
        The join method to use between the tile ids in the grid and the df of stats.
    stats : list or str
        The stats to be compute at the tile level on the `stat_col` column.
    
    Returns
    -------
    stats : pd.DataFrame
        The dataframe containing the tile id as index and with the stats in the
        `stat_col_{min/max/mean}` format.
        
    Examples
    --------
    >>> # Compute the per area buffered users based on home location (500m buffer):
    >>> users_buffered_per_area = mk.stats.computeBufferStat(
                                    gdf_stat=df_hw_locs_pd.reset_index()[['lat_home','lng_home', uidColName]],
                                    gdf_grid=gdf_aoi_grid,
                                    column=uidColName,
                                    aggregation=set,
                                    lat_name='lat_home',
                                    lon_name='lng_home',
                                    local_EPSG=local_EPSG,
                                    buffer=500)
    >>> # Compute the per area total daily traveled distance
    >>> ttd_daily_user = mk.spatial.totalUserTravelDistance(df_pings, freq='1d')
    >>> df_out = mk.stats.userBasedBufferedStat(ttd_daily_user,
                                                users_buffered_per_area,
                                                stat_col='ttd')
    >>> df_out.head()
    tile_ID   |    ttd_min |    ttd_max |   ttd_mean |    ttd_std |  ttd_count |
    12345     |      2.345 |     12.345 |      5.345 |      3.345 |        125 |
    
    '''
    if explode_col:
        df_stat = df_stat.explode(stat_col)
    stats_out = pd.merge(
            df_user_grid.reset_index().explode(uid_col)[[uid_col,tile_col]],
            df_stat.reset_index()[[uid_col,stat_col]],
            on=uid_col,
            how=how).groupby(tile_col).agg({stat_col: stats})
    if isinstance(stats, list):
        stats_out.columns = ["_".join(c) for c in stats_out.columns]
    return stats_out

def computeBufferStat(gdf_stat, gdf_grid,
                      column, aggregation,
                      how='inner',
                      lat_name='lat',
                      lon_name='lng',
                      local_EPSG=None,
                      buffer=None):
    '''
    Computes the statistics contained in a column of a dataframe containing the lat and lon
    coordinates of points with respect to a `gdf_grid` tessellation, possibly applying local
    reprojection and buffer on the points. This is equivalent to a KDE with a flat circular kernel of
    radius `buffer`.

    Parameters
    ----------
    gdf_stat, gdf_grid : gpd.GeoDataFrame
        The geo-dataframes containing the statistics in the `column` column and the tessellation system.
        They must be in the same reference system and will be projected to `local_EPSG`, if specified.
        The `gdf_grid` will be dissolved on the :attr:`mobilkit.dask_schemas.zidColName` after the spatial
        join with the (possibly buffered) `gdf_stat` geometries.
    column : str
        The column for which we will compute the statistics.
    aggregation : str or callable
        The geopandas string or callable to use on the spatially joined geo-dataframe.
    how : str, optional
        The method to perform the spatial join.
    lat_name, lon_name : str, optional
        The name of the columns to use as initial coords.
    local_EPSG : int, optional
        The code of the local EPSG crs.
    buffer : float, optional
        The local map unit in `local_EPSG` to perform the buffer.

    Returns
    -------
    buffered_stats : gpd.GeoDataFrame
        The geodataframe with the aggregated stat.
    '''
    if local_EPSG is not None:
        gdf_local = convert_df_crs(gdf_stat,
                                    lat_col=lat_name,
                                    lon_col=lon_name,
                                    from_crs='EPSG:4326',
                                    to_crs='EPSG:%d'%local_EPSG,
                                    return_gdf=True)
        gdf_grid = gdf_grid.to_crs(local_EPSG)
    else:
        gdf_local = gdf_stat.copy(deep=True)
        
    stats = _buffer_stat(gdf_local, gdf_grid,
                         column=column,
                         aggregation=aggregation,
                         buffer=buffer)
    return stats

def _buffer_stat(gdf_stat, gdf_grid,
                 column, aggregation,
                 how='inner',
                 local_EPSG=None,
                 buffer=None):
    '''
    Aux function to compute statistics in a geodataframe with respect to a gdf_grid tessellation.

    Parameters
    ----------
    gdf_stat, gdf_grid : gpd.GeoDataFrame
        The geo-dataframes containing the statistics in the `column` column and the tessellation system.
        They must be in the same reference system and will be projected to `local_EPSG`, if specified.
        The `gdf_grid` will be dissolved on the :attr:`mobilkit.dask_schemas.zidColName` after the spatial
        join with the (possibly buffered) `gdf_stat` geometries.
    column : str
        The column for which we will compute the statistics.
    aggregation : str or callable
        The geopandas string or callable to use on the spatially joined geo-dataframe.
    how : str, optional
        The method to perform the spatial join.
    local_EPSG : int, optional
        The code of the local EPSG crs.
    buffer : float, optional
        The local map unit in `local_EPSG` to perform the buffer.

    Returns
    -------
    buffered_stats : gpd.GeoDataFrame
        The geodataframe with the aggregated stat.
    '''
    if local_EPSG is not None:
        gdf_stat = gdf_stat.to_crs(local_EPSG)
        gdf_grid = gdf_grid.to_crs(local_EPSG)
    if buffer is not None:
        gdf_stat['geometry'] = gdf_stat.buffer(buffer)
    
    buffered_stat = gpd.sjoin(
                        gdf_grid[[zidColName, 'geometry']],
                        gdf_stat[['geometry', column]],
                        how=how,
                ).dissolve(zidColName, aggfunc=aggregation)
    return buffered_stat

def computeUserHomeWorkTripTimes(df_hw_locs,
                                 osrm_url=None,
                                 direction='both',
                                 what='duration',
                                 max_trip_duration_h=4,
                                 max_trip_distance_km=150):
    '''
    **TODO**
    This is quite slow as it is a serial part, it can be parallelized using a pool or directly
    mapping in Dask
    Returns
    -------
    time in seconds, distance in meters
    '''
    print('Computing straight hw distance...')
    df_hw_locs['home_work_straight_dist'] = df_hw_locs.apply(userHomeWorkDistance, axis=1)
    if osrm_url is not None:
        directions = []
        if direction in ['hw', 'both']:
            directions.append('hw')
        if direction in ['wh', 'both']:
            directions.append('wh')

        for direction in directions:
            tmp_str_direction = 'home_work_osrm' if direction == 'hw' else 'work_home_osrm'
            print('Computing %s OSM %s...' % (direction, what))
            tmp_res = df_hw_locs.apply(userHomeWorkTravelTimeOSRM,
                                       axis=1,
                                       osrm_url=osrm_url,
                                       direction=direction,
                                       what=what,
                                       max_trip_duration_h=max_trip_duration_h,
                                       max_trip_distance_km=max_trip_distance_km,
                                    )
            if what == 'distance':
                tmp_col = tmp_str_direction + "_" + 'dist'
                df_hw_locs[tmp_col] = tmp_res
            elif what == 'duration':
                tmp_col = tmp_str_direction + "_" + 'time'
                df_hw_locs[tmp_col] = tmp_res
            elif what == 'duration,distance':
                tmp_col = tmp_str_direction + "_" + 'time'
                df_hw_locs[tmp_col] = tmp_res.apply(lambda t: t[0])
                tmp_col = tmp_str_direction + "_" + 'dist'
                df_hw_locs[tmp_col] = tmp_res.apply(lambda t: t[1])
    return df_hw_locs


def _per_user_real_home_work_times(g,
                              direction='hw',
                              min_duration_h=.25,
                              max_duration_h=4.):
    if direction in ['hw', 'wh']:
        directions = [direction,]
    elif direction == 'both':
        directions = ['hw', 'wh']
    else:
        raise RuntimeError('Unknown direction %s in userHomeWorkTimes' % direction)
    
    df = g.sort_values(dttColName).copy(deep=True)
    df['atHome'] = df[zidColName] == df['home_'+zidColName]
    df['atWork'] = df[zidColName] == df['work_'+zidColName]
    df = df[df['atHome'] | df['atWork']].copy(deep=True)
    
    tmp_fields = {what+'_trips_'+k: []
                        for k in directions
                            for what in ['time','speed']}
    # Straight dist
    try:
        tmp_home_work_dist = g.iloc[0]['home_work_straight_dist']
    except KeyError:
        pass
    else:
        tmp_fields['home_work_straight_dist'] = tmp_home_work_dist
        
    # OSRM time
    try:
        tmp_home_work_travel = g.iloc[0]['home_work_osrm_time']
    except KeyError:
        pass
    else:
        tmp_fields['home_work_osrm_time'] = tmp_home_work_travel
        
    try:
        tmp_work_home_travel = g.iloc[0]['work_home_osrm_time']
    except KeyError:
        pass
    else:
        tmp_fields['work_home_osrm_time'] = tmp_work_home_travel
    
    # OSRM dist
    try:
        tmp_home_work_osrm_dist = g.iloc[0]['home_work_osrm_dist']
    except KeyError:
        pass
    else:
        tmp_fields['home_work_osrm_dist'] = tmp_home_work_osrm_dist
        
    try:
        tmp_work_home_osrm_dist = g.iloc[0]['work_home_osrm_dist']
    except KeyError:
        pass
    else:
        tmp_fields['work_home_osrm_dist'] = tmp_work_home_osrm_dist
        
    time_trips = pd.Series(tmp_fields)
    if df.shape[0] < 2 | df['atHome'].sum() == 0 | df['atHome'].sum():
        pass
    else:
        for direction in directions:
            if direction == 'hw':
                colShifted = 'atWork'
                colReference = 'atHome'
                df['next_location'] = df['atWork'].shift(-1)
            elif direction == 'wh':
                colShifted = 'atHome'
                colReference = 'atWork'
            else:
                raise RuntimeError('Unknown direction %s in userHomeWorkTimes' % direction)
            df['next_location'] = df[colShifted].shift(-1)
            df['next_start'] = df[dttColName].shift(-1)

            for _, r in df.iterrows():
                if r[colReference] == True and r['next_location'] == True:
                    tmp_duration_h = (r['next_start'] - r['leaving_datetime']).total_seconds() / 3600
                    if tmp_duration_h >= min_duration_h and tmp_duration_h <= max_duration_h:
                        time_trips.loc['time_trips_'+direction].append(tmp_duration_h)
                        time_trips.loc['speed_trips_'+direction].append(tmp_home_work_dist / tmp_duration_h)
    return time_trips

def userRealHomeWorkTimes(df_stops, direction='both', **kwargs):
    meta_out = {}
    if direction in ['hw', 'both']:
        meta_out['time_trips_hw'] = object
        meta_out['speed_trips_hw'] = object
        
    if direction in ['wh', 'both']:
        meta_out['time_trips_wh'] = object
        meta_out['speed_trips_wh'] = object

    cols_to_check = {
         'home_work_straight_dist': float,
         'home_work_osrm_time': float,
         'work_home_osrm_time': float,
         'home_work_osrm_dist': float,
         'work_home_osrm_dist': float,
     }
    for k, v in cols_to_check.items():
        if k in df_stops.columns:
            meta_out[k] = v
            
    user_time_trips_hw = df_stops.groupby(uidColName).apply(_per_user_real_home_work_times,
                                                            meta=meta_out,
                                                            direction=direction,
                                                            **kwargs
                                                           )
    return user_time_trips_hw

# Buffer the users and compute average and std trip time for each grid cell
def computeTripTimeStats(df_trip_times,
                         df_hw_locs,
                         gdf_grid,
                         local_EPSG,
                         buffer_m=500):
    print('Reprojecting grid...')
    gdf_grid_local = gdf_grid.to_crs(local_EPSG)
    print('Reprojecting homes...')
    gdf_local_homes = convert_df_crs(df_hw_locs.reset_index()[[uidColName,'lat_home','lng_home']],
                                     lon_col='lng_home',
                                     lat_col='lat_home',
                                     to_crs=local_EPSG,
                                     return_gdf=True,
                                     )

    print('Buffering and sjoin homes...')
#     gdf_local_homes['geometry'] = gdf_local_homes.buffer(buffer_m)
#     # gdf_local_works['geometry'] = gdf_local_works.buffer(buffer_m)
    
#     print('Spatial join of buffered homes and grid...')
#     user_per_home_area = gpd.sjoin(
#                     gdf_grid_local[[zidColName, 'geometry']],
#                     gdf_local_homes[['geometry',uidColName]],
#                 ).dissolve(zidColName, aggfunc=set)
    user_per_home_area = _buffer_stat(gdf_stat=gdf_local_homes,
                                      gdf_grid=gdf_grid_local,
                                      buffer=buffer_m,
                                      column=uidColName,
                                      aggregation=set,
                                      how='inner')
    
    print('Computing per area trip duration stats...')
    cols_to_do = ['time_trips_hw', 'time_trips_wh',
                'speed_trips_hw', 'speed_trips_wh',
                'home_work_straight_dist',
                'home_work_osrm_time',
                'work_home_osrm_time',
                'home_work_osrm_dist',
                'work_home_osrm_dist',
               ]
    cols_to_explode = ['time_trips_hw', 'time_trips_wh',
                       'speed_trips_hw', 'speed_trips_wh']
    for col in cols_to_do:
        if col not in df_trip_times.columns:
            continue
        for what in ['min','max','avg','std']:
            user_per_home_area[col+'_'+what] = None
            
    for idx, row in user_per_home_area.iterrows():
        tmp_users = row[uidColName].intersection(df_trip_times.index)
        if len(tmp_users) == 0:
            continue
        for col in cols_to_do:
            if col not in df_trip_times.columns:
                continue
            tmp_vals = df_trip_times.loc[tmp_users][col]
            if col in cols_to_explode:
                tmp_vals = tmp_vals.explode()
            tmp_vals = tmp_vals.values
            tmp_vals = tmp_vals[~pd.isna(tmp_vals)]
            if len(tmp_vals) > 0:
                for what_str, what_op in zip(['min','max','avg','std'],
                                             [np.min, np.max, np.mean,np.std]):
                    user_per_home_area.loc[idx,col+'_'+what_str] = what_op(tmp_vals)
    del user_per_home_area['geometry']
    del user_per_home_area['index_right']
    
    return user_per_home_area


def plotUsersHist(users_stats, min_pings=5, min_days=5, days="active", cmap='YlGnBu', xbins=100, ybins=20):
    '''Function to plot the 2d histogram of the users stats.
    
    Parameters
    ----------
    users_stats : pandas.DataFrame
        A dataframe with the users stats as returned by :attr:`mobilkit.stats.userStats` and passed to pandas with the ``toPandas`` method.
    min_pings : int, optional
        The number of pings to be used as threshold in the plot counts.
    min_days : int, optional
        The number of active or spanned days (depending on ``days``) to be used as threshold in the plot counts.
    days : str, optional
        Whether to use active (``active``, default) days or spanned days (``spanned``).
    cmap : str, optional
        The colormap to use.
    xbins, ybins : int, optional
        The number of bins to use on the x and y axis.

    Returns
    -------
    ax
        The axes of the figure.
    '''

    assert days in ["active", "spanned"]
    # Plot the 2dhistogram

    if days == "active":
        col_days = "daysActive"
    else:
        col_days = "daysSpanned"

    df_users_stats = users_stats[["uid","pings",col_days]]

    XXX = np.array(df_users_stats["pings"].values, dtype="float")
    YYY = np.array(df_users_stats[col_days].values, dtype="float")

    YYY = YYY[XXX>0]
    XXX = XXX[XXX>0]

    XXX = np.log10(XXX)

    min_x = np.log10(min_pings)
    min_y = min_days

    min_count = XXX.min()
    max_count = XXX.max()*1.02

    min_days = max(1, YYY.min())
    max_days = YYY.max() + 1

    x_grid = np.linspace(min_count, max_count, xbins)
    y_grid = np.linspace(min_days, max_days, ybins)

    ul = np.logical_and(YYY>=min_y,XXX<min_x).sum()
    ur = np.logical_and(YYY>=min_y,XXX>=min_x).sum()
    lr = np.logical_and(YYY<min_y,XXX>=min_x).sum()
    ll = np.logical_and(YYY<min_y,XXX<min_x).sum()

    grid, _, __ = np.histogram2d(XXX, YYY,
                                bins=[x_grid, y_grid])

    grid = grid.T

    fig, ax = plt.subplots(1,1,figsize=(12,10))

    plt.title("ul: %d - ur: %d - lr: %d - ll: %d" % (ul, ur, lr, ll), size=26)

    X, Y = np.meshgrid(x_grid, y_grid)
    plt.pcolormesh(X, Y, grid, cmap='YlGnBu',
                   # norm=matplotlib.colors.LogNorm())
                   # norm=matplotlib.colors.LogNorm(vmin=1, vmax=grid.max()))
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=grid.max()))
    cbar = plt.colorbar()

    plt.vlines(min_x, min_days, max_days, linestyles="--", color="r", lw=2)
    plt.hlines(min_y, min_count, max_count, linestyles="--", color="r", lw=2)


    cbar.set_label('#user', labelpad=-40, y=1.05, rotation=0, fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    ax.set_yticks(np.arange(0,max_days,30))
    ax.axis([0, 9, 1, 365])
    ax.set_xlabel('log10 #records', size=24)
    ax.set_ylabel('timespan (days %s)' % col_days, size=24)
    ax.tick_params(labelsize=20)

    plt.xlim(0, max_count)
    plt.ylim(0, max_days)

    plt.tight_layout()

    return ax

def computeSurvivalFracs(users_stats_df, thresholds=[1,10,20,50,100]):
    '''
    Function to compute the fraction of users above threshold.
    
    Parameters
    ----------
    users_stats : pandas.DataFrame
        A dataframe with the users stats as returned by :attr:`mobilkit.stats.userStats` and passed to pandas with the ``toPandas`` method.
    thresholds : list or array of ints, optional
        The values of the threshold to compute. The number of days above the threshold and the fraction of active days above threshold will be saved, for each user, in the ``days_above_TTT`` and ``frac_days_above_TTT`` where ``TTT`` is the threshold value.
    Returns
    -------
    df : pandas.DataFrame
        The enriched dataframe.
    '''
    
    for thr in thresholds:
        users_stats_df["days_above_%03d"%thr] = \
            users_stats_df["pingsPerDay"].apply(lambda v:
                                            sum([d>thr for d in v]))
        users_stats_df["frac_days_above_%03d"%thr] = \
                    users_stats_df["days_above_%03d"%thr]\
                        / users_stats_df["daysActive"]
        
    return users_stats_df
    

def plotSurvivalDays(users_stats_df, min_days=10, ax=None):
    '''Function to plot the survival probability of users by number of days given different pings/day threshold.
    
    Parameters
    ----------
    users_stats_df : pandas.DataFrame
        A dataframe with the users stats as returned by :attr:`mobilkit.stats.computeSurvivalFracs`.
    min_days : int, optional
        The minimum number of active days above threshold to be counted as super user in the plot count.
    ax : plt.axes, optional
        The axes to use. If ``None`` a new figure will be produced.

    Returns
    -------
    ax
        The axes of the figure.
    '''
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))

    for colname in sorted(users_stats_df.columns):
        if not colname.startswith("days_above_"):
            continue
        else:
            thr = int(colname.split("days_above_")[1])
            
        col = users_stats_df[colname]
        n_over = sum(col>=min_days)

        bins = np.linspace(1, max(col)+1, 40)

        f, b = np.histogram(col, bins=bins, density=True)
        ccdf = 1. - np.cumsum(f) * np.diff(b)

        b_center = (b[1:]+b[:-1])/2.
        plt.plot(b_center, ccdf, ".-", label="thr: %d - users: %.02e" % (thr, n_over))

    plt.vlines(min_days, 1e-8, 1., colors="k", alpha=.8, linestyles="--", lw=2)
    plt.xlabel(r"Days with pings >= threshold - $d$", size=16)
    plt.ylabel(r"$CCDF(d)$", size=16)

    plt.semilogy()
    plt.legend()
    
    return ax

def plotSurvivalFrac(users_stats_df, min_frac=.8, ax=None):
    '''Function to plot the survival probability of users by fraction of active days given different pings/day threshold.
    
    Parameters
    ----------
    users_stats_df : pandas.DataFrame
        A dataframe with the users stats as returned by :attr:`mobilkit.stats.computeSurvivalFracs`.
    min_frac : 0 < float < 1, optional
        The minimum fraction of active days above threshold to be counted as super user in the plot count.
    ax : plt.axes, optional
        The axes to use. If ``None`` a new figure will be produced.

    Returns
    -------
    ax
        The axes of the figure.
    '''
    
    assert 0 < min_frac < 1
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(6,4))

    for colname in sorted(users_stats_df.columns):
        if not colname.startswith("frac_days_above_"):
            continue
        else:
            thr = int(colname.split("frac_days_above_")[1])
            
        col = users_stats_df[colname]
        n_over = sum(col>=min_frac)

        bins = np.linspace(0, 1., 40)

        f, b = np.histogram(col, bins=bins, density=True)
        ccdf = 1. - np.cumsum(f) * np.diff(b)

        b_center = (b[1:]+b[:-1])/2.
        plt.plot(b_center, ccdf, ".-", label="thr: %d - users: %.02e" % (thr, n_over))

    plt.vlines(min_frac, 1e-4, 1., colors="k", alpha=.8, linestyles="--", lw=2)
    plt.xlabel(r"Fraction of active days with pings >= threshold - $d$", size=16)
    plt.ylabel(r"$CCDF(d)$", size=16)

    plt.semilogy()
    plt.legend()
    
    return ax

