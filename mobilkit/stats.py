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
from mobilkit.spatial import convert_df_crs, userHomeWorkDistance, haversine_pairwise
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
    ldtColName,
    durColName,
    zidColName,
    medLatColName,
    medLonColName,
    locColName,
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

'''There are two possible approaches to the home computation
  The one here leverages on stops and/or locations.
  Another one is below, and uses the raw pings and the tessellated areas only.
'''

def stopsToHomeWorkStats(df_stops,
                         home_hours=(21,7),
                         work_hours=(9,17),
                         work_days=(0,1,2,3,4),
                         force_different=False,
                         ignore_dynamical=True,
                         min_hw_distance_km=.0,
                         min_home_delta_count=0,
                         min_home_delta_duration=0,
                         min_work_delta_count=0,
                         min_work_delta_duration=0,
                         min_home_days=0,
                         min_work_days=0,
                         min_home_hours=0,
                         min_work_hours=0):
    '''
    Computes the home and work time stats for each user and location (tile).

    Parameters
    ----------
    df_stop_locs_usr : dask.DataFrame or pd.DataFrame
        The stops of a user as returned by locations or stops TODO;
    home_hours, work_hours : tuple, optional
        TODO
    work_days : tuple
        TODO
    force_different : bool, optional
        TODO
    ignore_dynamical : bool, optional
        TODO
    min_hw_distance_km : float, optional
        TODO
    min_home_delta_count, min_home_delta_duration,
    min_work_delta_count, min_work_delta_duration : float, optional
        TODO
    min_home_days, min_work_days,
    min_home_hours, min_work_hours :  int, optional
        TODO
    latCol, lonCol, locCol : str, optional
        TODO
        
    Returns
    -------
    df_stats : pd.DataFrame
        A dataframe with the columns:
        - `uid` the user id
        - 'loc_id' or 'tile_ID' the location/tile id 0-based;
        - 'lat_medoid','lng_medoid' or 'lat', 'lng' the average coordinates of the stops
          seen within that location/tile;
        - '{home,work}_{day/hour}_count' the number of unique days (hours) when the user has
          been seen as active in the location (tile) at home (work) hours;
       - '{home,work}_per_hour_{count,duration}' the list containing, for each hour in the home (work)
         hours, the number of visits (duration in seconds) spent at the location/tile;
       - '{home,work}_{count,duration}' the total number of visits (seconds duration) spent at this
         location/tile;
       - 'tot_seen_{home,work}_{hours,days}' the total number of days and hours where the user has been
         active during home (work) hours during the valid stops;
       - 'tot_seen_{hours,days}' the total number of days and hours where the user has been
         active during the valid stops, both in home and workj period;
       - 'tot_stop_count', 'tot_stop_time' the total number and duration (in seconds) of the user's
         stops;
       - 'frac_{home,work}_{count,duration}' the fraction of stops (duration) spent in this tile/location
         during home (work) hours;
       - '{home,work}_delta_{count,duration}' the fraction of hours in the home (work) range at which
         the given tile/location was the most visited in terms of stops (duration).
       - 'isHome', 'isWork' the flag telling whsther the location is home or work (or potentially both,
         if `force_different` is False).
    '''
    
    # Determines if we have a stops or locations dataframe
    if medLonColName in df_stops.columns and locColName in df_stops.columns:
        latCol = medLatColName
        lonCol = medLonColName
        locCol = locColName
    elif lonColName in df_stops.columns and zidColName in df_stops.columns:
        latCol = latColName
        lonCol = lonColName
        locCol = zidColName
        
    out_meta = {
        locCol: int, 
        latCol: float,
        lonCol: float,
        'home_day_count': int,
        'home_hour_count': int,
        'home_per_hour_count': object,
        'home_per_hour_duration': object,
        'work_day_count': int,
        'work_hour_count': int,
        'work_per_hour_count': object,
        'work_per_hour_duration': object,
        'home_count': int,
        'work_count': int,
        'home_duration': float,
        'work_duration': float,
        'tot_seen_home_hours': int,
        'tot_seen_home_days': int,
        'tot_seen_work_hours': int,
        'tot_seen_work_days': int,
        'tot_seen_hours': int,
        'tot_seen_days': int,
        'tot_stop_count': int,
        'tot_stop_time': float,
        'frac_home_count': float,
        'frac_work_count': float,
        'frac_home_duration': float,
        'frac_work_duration': float,
        'home_delta_count': float,
        'work_delta_count': float,
        'home_delta_duration': float,
        'work_delta_duration': float,
        'isHome': bool,
        'isWork': bool,
        uidColName: str,
    }
        
    df_stats = df_stops.groupby(uidColName)\
                  .apply(_compute_usr_hw_stats_locations,
                        home_hours=home_hours,
                        work_hours=work_hours,
                        work_days=work_days,
                        force_different=force_different,
                        ignore_dynamical=ignore_dynamical,
                        min_hw_distance_km=min_hw_distance_km,
                        min_home_delta_count=min_home_delta_count,
                        min_home_delta_duration=min_home_delta_duration,
                        min_work_delta_count=min_work_delta_count,
                        min_work_delta_duration=min_work_delta_duration,
                        min_home_days=min_home_days,
                        min_work_days=min_work_days,
                        min_home_hours=min_home_hours,
                        min_work_hours=min_work_hours,
                        latCol=latCol,
                        lonCol=lonCol,
                        locCol=locCol,
                        meta=out_meta,
                    ).reset_index(drop=True)
    
    return df_stats


def compressLocsStats2hwTable(df):
    '''
    Transforms a per location home work stats table into the per user home and work stats table.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing all the stats of the locations.

    Returns
    -------
    hw_stats : pd.DataFrame
        The home work locations of the users as if they were returned by
        :attr:`mobilkit.stats.userHomeWorkLocation`.
    '''
    return df.groupby(uidColName).apply(_compress_locs_stats_to_table)


def computeHomeWorkSurvival(df_stops_stats,
                            min_durations=[0],
                            min_day_counts=[0],
                            min_hour_counts=[0],
                            min_delta_counts=[0],
                            min_delta_durations=[0],
                            limit_hw_locs=False,
                            loc_col=locColName,
                      ):
    '''
    Given a dataframe of locations (tiles) with home work stats as returned by
    :attr:`mobilkit.stats.stopsToHomeWorkStats` it computes the home and work presence
    at different thresholds of home and work duration count etc.
    
    Parameters
    ----------
    df_stops_stats : pandas.DataFrame
        The locations (or tiles) stats of the users as returned by
        :attr:`mobilkit.stats.stopsToHomeWorkStats`.
    min_durations : iterable, optional
        The minimum duration of home and work stops to keep lines in the group.
    min_day_counts, min_hour_counts : iterable, optional
        The minimum count of stops in home and work locations to keep lines
        in the group.
    min_delta_counts, min_delta_durations : iterable, optional
        The minimum fraction of home/work hours during which the area/location is the
        most visited in terms of duration/count of stops for it to be kept.
    limit_hw_locs : bool, optional
        If `True`, it will limit the home and work candidates to the row(s) featuring
        `isHome` or `isWork` equal `True`, respectively. If `False` (default), all the
        rows are kept as candidates.
    loc_col : str, optional
        The column to use to check if the home and work candidates are in the same
        location.
    
    Returns
    -------
    user_flags : pd.DataFrame
        A data frame indexed by user containing, for each combination of threshold values
        in the order of minimum duration, minimum days, minimum hours, min delta count, min delta
        duration, the flag of:
        - `out_flags` if the user has a home AND work candidate with the threshold;
        - `out_has_home` if the user has a home candidate with the threshold;
        - `out_has_work` if the user has a work candidate with the threshold;
        - `out_same_locs` if the user has a unique the home and work candidate falling
          under the same `loc_col` ID.
    df_cnt : pd.DataFrame
        The dataframe in long format containing the count of valid counts for the users
        for each combination of minimum threshold.
        The columns are:
        - 'tot_duration', 'n_days', 'n_hours', 'delta_count', 'delta_duration'
          the values of the constraint for the current count.
        - 'n_users' how many users have both home and work with current settings;
        - 'with_home_users' how many users have a home location with current settings;
        - 'with_work_users' how many users have a work location with current settings;
        - 'home_work_same_area_users' how many users have home and work locations
          featuring the same `loc_col` ID.
        - 'home_work_same_area_users_frac' the fraction of valid users with home and work
          that have have home and work locations featuring the same `loc_col` ID.
    '''
    
    col_values = list(_cycle_dur_count_delta(
                            min_durations=min_durations,
                            min_day_counts=min_day_counts,
                            min_hour_counts=min_hour_counts,
                            min_delta_counts=min_delta_counts,
                            min_delta_durations=min_delta_durations,
                    ))

    out_df = df_stops_stats.groupby(uidColName).apply(
                                            _keep_flag_hw_user,
                                            min_durations=min_durations,
                                            min_day_counts=min_day_counts,
                                            min_hour_counts=min_hour_counts,
                                            min_delta_counts=min_delta_counts,
                                            min_delta_durations=min_delta_durations,
                                            loc_col=loc_col,
                                            limit_hw_locs=limit_hw_locs
                                             )
    
    count_users = np.stack(out_df['out_flags']).sum(axis=0)
    with_home_users = np.stack(out_df['out_has_home']).sum(axis=0)
    with_work_users = np.stack(out_df['out_has_work']).sum(axis=0)
    howo_same_users = np.stack(out_df['out_same_locs']).sum(axis=0)

    df_cnt = []
    for cnt, hasH, hasW, sameA, vals in zip(count_users,
                                           with_home_users,
                                           with_work_users,
                                           howo_same_users,
                                           col_values):
        df_cnt.append([vals[i] for i in range(len(vals))] + [cnt, hasH, hasW, sameA])
    df_cnt = pd.DataFrame(df_cnt, columns=[
        'tot_duration', 'n_days', 'n_hours', 'delta_count', 'delta_duration',
        'n_users', 'with_home_users', 'with_work_users', 'home_work_same_area_users',
    ])
    
    df_cnt['home_work_same_area_users_frac'] = df_cnt['home_work_same_area_users'] / df_cnt['n_users'].clip(lower=1)
    
    return out_df, df_cnt


'''These are the functions to only use pings and tessellation
'''
def userHomeWork(df, homeHours=(19.5,7.5), workHours=(9.,18.5), weHome=False):
    '''
    Computes, for each row of the dataset, if the ping has been recorded in home or
    work time. Can be used in combination with :attr:`mobilkit.stats.homeWorkStats'
    and :attr:'mobilkit.stats.userHomeWorkLocation` to determine the home and work
    locations of a user.

    Parameters
    ----------
    df : dask.dataframe
        The loaded dataframe with at least `uid`, `datetime` and `tile_ID` columns.
 
    homeHours :  tuple, optional
        The starting and end hours of the home period in 24h floating numbers. For example, to put the house
        period from 08:15pm to 07:20am put ``homeHours=(20.25, 7.33)``.
    workHours :  tuple, optional
        The starting and end hours of the work period in 24h floating numbers. For example, to put the work
        period from 09:15am to 06:50pm put ``workHours=(9.25, 18.8333)``.
        **Note that work hours are counted only from Monday to Friday.**
    weHome : bool, optional
        If ``False`` (default) counts only weekend hours within the home hours as valid home hours.
        If ``True``, all the pings recorded during the weekend (Saturday and Sunday) are counted as home pings.

    Returns
    -------
    out : dask.dataframe
        The dataframe with two additional columns: `isHome` and `isWork` telling if a given ping has been
        recorded during home or work time (or none of them).
    
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_,
    obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to
    preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the
    one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's
    home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to
    the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added
    to data in the user's home location area. Please check if your data has such noise added and choose the
    spatial tessellation according to your use case.
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
    '''Given a dataframe returned by :attr:`mobilkit.stats.userHomeWork` computes, for each user,
    the home and work area as well as their location.
    The home/work area is the one with more pings recorded and the location is assigned to the mean
    point of this cloud.

    Parameters
    ----------

    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at least the `uid`,
        `tile_ID` and `isHome` and `isWork` columns.
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
    When determining the home location of a user, please consider that some data providers,
    like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's
    home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher
    than the one used to obfuscate these data. If you are interested in the census area (or geohash)
    of the user's home alone and you are using a spatial tessellation with a spatial resolution wider
    than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise
    added to data in the user's home location area. Please check if your data has such noise added
    and choose the spatial tessellation according to your use case.
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
                              location_col=zidColName,
                              home_loc_col='home_'+zidColName,
                              work_loc_col='work_'+zidColName,
                              min_duration_h=.25,
                              max_duration_h=4.):
    if direction in ['hw', 'wh']:
        directions = [direction,]
    elif direction == 'both':
        directions = ['hw', 'wh']
    else:
        raise RuntimeError('Unknown direction %s in userHomeWorkTimes' % direction)
    
    df = g.sort_values(dttColName).copy()
    df['atHome'] = df[location_col] == df[home_loc_col]
    df['atWork'] = df[location_col] == df[work_loc_col]
    df = df[df['atHome'] | df['atWork']].copy()
    
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

def userRealHomeWorkTimes(df_stops,
                          home_work_locs, 
                          direction='both',
                          uid_col=uidColName,
                          location_col=zidColName,
                          additional_hw_cols=['home_work_straight_dist',
                                              'home_work_osrm_time',
                                              'work_home_osrm_time',
                                              'home_work_osrm_dist',
                                              'work_home_osrm_dist'],
                          **kwargs):
    '''
    Computes the real homework commuting time looking at the sequence of the user's stops.

    Parameters
    ----------
    df_stops : dd.DataFrame
        A dask dataframe containing the stops (or pings) of the users to be analyzed.
        It might feature a location id column (as when returned by
        :attr:`mobilkit.spatial.computeUsersLocations` applied on the output of
        :attr:`mobilkit.spatial.findStops`) or the tile id column (as in the df returned
        by :attr:`mobilkit.spatial.tessellate`).
    home_work_locs : dd.DataFrame or pd.DataFrame
        The necessarily pre-cleaned dataframe containing the stats on users home and work.
        This can be either a dataframe as returned by :attr:`mobilkit.stats.userHomeWorkLocation`
        or the one obtained by chaining the :attr:`mobilkit.stats.stopsToHomeWorkStats`
        and the :attr:`mobilkit.stats.`
    **kwargs 
        Are used to tune the functioning of :attr:`mobilkit.stats._per_user_real_home_work_times`

    Returns
    -------

    
    '''
    home_id_col = 'home_'+location_col
    work_id_col = 'work_'+location_col
    hw_cols_to_keep = [uid_col, home_id_col, work_id_col] + additional_hw_cols

    stops_users_home_work = df_stops.merge(home_work_locs.reset_index()[hw_cols_to_keep],
                                                how='inner',
                                                on=uid_col)
    stops_users_home_work = stops_users_home_work[
        (stops_users_home_work[location_col] == stops_users_home_work[home_id_col])
        | (stops_users_home_work[location_col] == stops_users_home_work[work_id_col])
    ].copy().repartition(npartitions=200)
    
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
        if k in stops_users_home_work.columns:
            meta_out[k] = v
            
    user_time_trips_hw = stops_users_home_work.groupby(uid_col)\
                                              .apply(
                                                _per_user_real_home_work_times,
                                                meta=meta_out,
                                                location_col=location_col,
                                                direction=direction,
                                                home_loc_col=home_id_col,
                                                work_loc_col=work_id_col,
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

'''**Auxiliary functions**
'''

def _compute_usr_hw_stats_locations(df_stop_locs_usr,
                                    home_hours=(21,7),
                                    work_hours=(9,17),
                                    work_days=(0,1,2,3,4),
                                    force_different=False,
                                    ignore_dynamical=True,
                                    min_hw_distance_km=.0,
                                    min_home_delta_count=0,
                                    min_home_delta_duration=0,
                                    min_work_delta_count=0,
                                    min_work_delta_duration=0,
                                    min_home_days=0,
                                    min_work_days=0,
                                    min_home_hours=0,
                                    min_work_hours=0,
                                    latCol=medLatColName,
                                    lonCol=medLonColName,
                                    locCol=locColName,
                                   ):
    '''
    Helper function to compute home and work locations from stops with or without
    locations annotations.

    Parameters
    ----------
    df_stop_locs_usr : dask.DataFrame or pd.DataFrame
        The stops of a user as returned by locations or stops TODO;
    home_hours, work_hours : tuple, optional
        TODO
    work_days : tuple
        TODO
    force_different : bool, optional
        TODO
    ignore_dynamical : bool, optional
        TODO
    min_hw_distance_km : float, optional
        TODO
    min_home_delta_count, min_home_delta_duration,
    min_work_delta_count, min_work_delta_duration : float, optional
        TODO
    min_home_days, min_work_days,
    min_home_hours, min_work_hours :  int, optional
        TODO
    latCol, lonCol, locCol : str, optional
        TODO
        
    Returns
    -------
    df_stats : pd.DataFrame
        A dataframe with the columns:
        - 'loc_id' or 'tile_ID' the location/tile id 0-based;
        - 'lat_medoid','lng_medoid' or 'lat', 'lng' the average coordinates of the stops
          seen within that location/tile;
        - '{home,work}_{day/hour}_count' the number of unique days (hours) when the user has
          been seen as active in the location (tile) at home (work) hours;
       - '{home,work}_per_hour_{count,duration}' the list containing, for each hour in the home (work)
         hours, the number of visits (duration in seconds) spent at the location/tile;
       - '{home,work}_{count,duration}' the total number of visits (seconds duration) spent at this
         location/tile;
       - 'tot_seen_{home,work}_{hours,days}' the total number of days and hours where the user has been
         active during home (work) hours during the valid stops;
       - 'tot_seen_{hours,days}' the total number of days and hours where the user has been
         active during the valid stops, both in home and workj period;
       - 'tot_stop_count', 'tot_stop_time' the total number and duration (in seconds) of the user's
         stops;
       - 'frac_{home,work}_{count,duration}' the fraction of stops (duration) spent in this tile/location
         during home (work) hours;
       - '{home,work}_delta_{count,duration}' the fraction of hours in the home (work) range at which
         the given tile/location was the most visited in terms of stops (duration).
       - 'isHome', 'isWork' the flag telling whsther the location is home or work (or potentially both,
         if `force_different` is False).
    '''
    # Save user id
    tmp_uid = df_stop_locs_usr[uidColName].iloc[0]
    # Prepare the containers of results:
    # - I compute the hours in home and work ranges
    reverseHomeHours = home_hours[0] > home_hours [1]
    reverseWorkHours = work_hours[0] > work_hours [1]
    # - I prepare the sets where to save the seend days and hours
    #   in the day, night and total case
    total_seen_days = set()
    total_seen_hours = set()
    total_seen_home_days = set()
    total_seen_home_hours = set()
    total_seen_work_days = set()
    total_seen_work_hours = set()
    # - The lists containing the int hours and counter to be populated
    if reverseHomeHours:
        home_hours_list = [h%24 for h in range(
                                    int(np.floor(home_hours[0])),
                                    int(np.ceil(home_hours[1])+24.))]
    else:
        home_hours_list = [h%24 for h in range(
                                    int(np.floor(home_hours[0])),
                                    int(np.ceil(home_hours[1])))]
    if reverseWorkHours:
        work_hours_list = [h%24 for h in range(
                                    int(np.floor(work_hours[0])),
                                    int(np.ceil(work_hours[1])+24.))]
    else:
        work_hours_list = [h%24 for h in range(
                                    int(np.floor(work_hours[0])),
                                    int(np.ceil(work_hours[1])))]
    # # Determines if we have a stops or locations dataframe
    # if medLonColName in df_stop_locs_usr.columns and locColName in df_stop_locs_usr.columns:
    #     latCol = medLatColName
    #     lonCol = medLonColName
    #     locCol = locColName
    # elif lonColName in df_stop_locs_usr.columns and zidColName in df_stop_locs_usr.columns:
    #     latCol = latColName
    #     lonCol = lonColName
    #     locCol = zidColName
    # If ignoring the non assigned locations, we prune them here
    if ignore_dynamical:
        df_stop_locs_usr = df_stop_locs_usr.query(f'{locCol} >= 0')
        
    if df_stop_locs_usr.shape[0] == 0:
        return None
    
    # Prepare the dictionary for each location
    locs_stats = {k: {
            locCol: k,
            latCol: latlon[latCol],
            lonCol: latlon[lonCol],
            'home_day_count': set(),
            'home_hour_count': set(),
            'home_per_hour_count': np.array([0 for _ in home_hours_list]),
            'home_per_hour_duration': np.array([.0 for _ in home_hours_list]),
            'work_day_count': set(),
            'work_hour_count': set(),
            'work_per_hour_count': np.array([0 for _ in work_hours_list]),
            'work_per_hour_duration': np.array([.0 for _ in work_hours_list]),
            }
        for k, latlon in df_stop_locs_usr.groupby(locCol)[[latCol,lonCol]].mean().iterrows()}
    # Cycle over the stops to populate the duration and count of visits:
    for _, stp_loc in df_stop_locs_usr.iterrows():
        # - Extract location id and make reference to tmp target dict
        tmp_loc = stp_loc[locCol]
        tmp_loc_dict = locs_stats[tmp_loc]
        last_hour = None
        for t in pd.date_range(stp_loc[dttColName], stp_loc[ldtColName], freq='1min'):
            # Cycle over all the minutes to:
            # - compute the uniuque hour and day identifiers, the float/int hour and the dow
            tmp_unique_h = t.strftime('%Y%m%d%H')
            tmp_unique_d = tmp_unique_h[:-2]
            total_seen_hours.add(tmp_unique_h)
            total_seen_days.add(tmp_unique_d)
            tmp_h = t.hour + t.minute/60.
            tmp_h_int = t.hour
            tmp_dow = t.dayofweek
            # Check if hour is in home range and, if new hour, add it to the set of home
            # hours days
            isHome = False
            if reverseHomeHours:
                if not home_hours[1] <= tmp_h < home_hours[0]:
                    isHome = True
            elif home_hours[0] <= tmp_h < home_hours[1]:
                isHome = True
            if isHome:
                # If home, append seen hour and day to the location's and total sets 
                tmp_loc_dict['home_day_count'].add(tmp_unique_d)
                tmp_loc_dict['home_hour_count'].add(tmp_unique_h)
                total_seen_home_hours.add(tmp_unique_h)
                total_seen_home_days.add(tmp_unique_d)
                # Also add one to the visits if this is a new hour and 60 secs to the
                # duration in the corresponding hour's list item
                home_hour_index = home_hours_list.index(tmp_h_int)
                if last_hour != tmp_h_int:
                    tmp_loc_dict['home_per_hour_count'][home_hour_index] += 1
                tmp_loc_dict['home_per_hour_duration'][home_hour_index] += 60
            elif tmp_dow in work_days:
                # Enter the work if we're in the right days and repeat the same logic
                # of home
                isWork = False
                if reverseWorkHours:
                    if not work_hours[1] <= tmp_h < work_hours[0]:
                        isWork = True
                elif work_hours[0] <= tmp_h < work_hours[1]:
                    isWork = True
                if isWork:
                    tmp_loc_dict['work_day_count'].add(tmp_unique_d)
                    tmp_loc_dict['work_hour_count'].add(tmp_unique_h)
                    total_seen_work_hours.add(tmp_unique_h)
                    total_seen_work_days.add(tmp_unique_d)
                    work_hour_index = work_hours_list.index(tmp_h_int)
                    if last_hour != tmp_h_int:
                        tmp_loc_dict['work_per_hour_count'][work_hour_index] += 1
                    tmp_loc_dict['work_per_hour_duration'][work_hour_index] += 60
            # Update last hour
            last_hour = tmp_h_int
    # At the end of the cycle transorm the sets and counter to len and sums
    for k, v in locs_stats.items():
        # How many visits we made to location in home/work hours
        v['home_count'] = sum(v['home_per_hour_count'])
        v['work_count'] = sum(v['work_per_hour_count'])
        # How many seconds we spent at location in home/work hours
        v['home_duration'] = sum(v['home_per_hour_duration'])
        v['work_duration'] = sum(v['work_per_hour_duration'])
        # How many unique days/hours we visited location in home/work hours
        v['home_day_count'] = len(v['home_day_count'])
        v['work_day_count'] = len(v['work_day_count'])
        v['home_hour_count'] = len(v['home_hour_count'])
        v['work_hour_count'] = len(v['work_hour_count'])
        
    # Port to dataframe and add global values: 
    df_stats = pd.DataFrame.from_dict(locs_stats, orient='index')
    # - How many unique hours/days we seen in total in home/work hours
    df_stats['tot_seen_home_hours'] = len(total_seen_home_hours)
    df_stats['tot_seen_home_days'] = len(total_seen_home_days)
    df_stats['tot_seen_work_hours'] = len(total_seen_work_hours)
    df_stats['tot_seen_work_days'] = len(total_seen_work_days)
    # - How many unique hours/days/stops/duration we seen in total
    df_stats['tot_seen_hours'] = len(total_seen_hours)
    df_stats['tot_seen_days'] = len(total_seen_days)
    df_stats['tot_stop_count'] = df_stop_locs_usr.shape[0]
    df_stats['tot_stop_time'] = df_stop_locs_usr[durColName].sum()
    # - The fraction of hours seen (duration spent) at location w.r.t. the total valid stops
    df_stats['frac_home_count'] = df_stats['home_count'] / df_stats['tot_seen_home_hours'].clip(lower=1.)
    df_stats['frac_work_count'] = df_stats['work_count'] / df_stats['tot_seen_work_hours'].clip(lower=1.)
    df_stats['frac_home_duration'] = df_stats['home_duration'] / max(1.,df_stats['home_duration'].sum())
    df_stats['frac_work_duration'] = df_stats['work_duration'] / max(1.,df_stats['work_duration'].sum())
    
    # - Compute the delta (fraction of hours for which the location is the most visited in
    # terms of counts/duration )
    for source, target in zip([
                        'home_per_hour_count', 'work_per_hour_count',
                        'home_per_hour_duration', 'work_per_hour_duration',], [
                        'home_delta_count', 'work_delta_count',
                        'home_delta_duration', 'work_delta_duration']):
        # Count the number of items in each row that are the maximum column wise
        # and then transform in the fraction 
        tmp_vs = np.stack(df_stats[source])
        tmp_vs = (tmp_vs == tmp_vs.max(axis=0, keepdims=True).clip(min=1.)).sum(axis=1, keepdims=True)\
                / max(1, tmp_vs.shape[1])
        df_stats[target] = tmp_vs.squeeze(-1,)
        
    # Find home which is the most visited during home hours respecting the parameters
    candidates =   df_stats.query(f"home_day_count >= {min_home_days} & home_hour_count >= {min_home_hours}")
    candidates = candidates.query(f"work_day_count >= {min_work_days} & work_hour_count >= {min_work_hours}")
    candidates = candidates.query(f"home_delta_count >= {min_home_delta_count} & work_delta_count >= {min_work_delta_count}")
    candidates = candidates.query(f"home_delta_duration >= {min_home_delta_duration} & work_delta_duration >= {min_work_delta_duration}")
    home_loc_id = None
    work_loc_id = None
    if candidates.shape[0] > 0:
        home_loc = candidates.sort_values('home_delta_duration', ascending=False).iloc[0]
        if home_loc['home_hour_count'] > 0:
            home_loc_id = home_loc[locCol]
        # Now the work location:
        # - filter the home if we want to enforce different locations
        if force_different:
            candidates = candidates.query(f'{locCol} != {home_loc_id}')
        if candidates.shape[0] > 0:
            # - filter the close locations if we want to enforce home work distance
            if min_hw_distance_km > 0:
                home_latlon = [[home_loc[latCol], home_loc[lonCol]]]
                candidates['hw_distance_km'] = haversine_pairwise(
                                                        candidates[[latCol,lonCol]].values,
                                                        home_latlon,
                                                    )
                candidates = candidates.query(f'hw_distance_km >= {min_hw_distance_km}')
            work_loc = candidates.sort_values('work_delta_duration', ascending=False).iloc[0]
            if work_loc['work_hour_count'] > 0:
                work_loc_id = work_loc[locCol]
    # Annotate the home/work results
    df_stats['isHome'] = df_stats[locCol].apply(lambda i: i==home_loc_id)
    df_stats['isWork'] = df_stats[locCol].apply(lambda i: i==work_loc_id)
    # df_stats[uidColName] = df_stop_locs_usr.iloc[0][uidColName]
    # df_stats = df_stats.set_index([locCol])
    df_stats[uidColName] = tmp_uid
    return df_stats



def _compress_locs_stats_to_table(g):
    '''
    Helper function to :attr:`mobilkit.stats.compressLocsStats2hwTable`
    to compute the home and work locations in the format of
    :attr:`mobilkit.stats.userHomeWorkLocation`.
    '''
    if locColName in g.columns:
        loc_col = locColName
        lat_col = medLatColName
        lon_col = medLonColName
    elif zidColName in g.columns:
        loc_col = zidColName
        lat_col = latColName
        lon_col = lonColName
    else:
        raise RuntimeError('No location or cell columns found in stop stats')
        
    home = g.query('isHome == True')
    work = g.query('isWork == True')
    tot_pings = g['tot_stop_count'].iloc[0]
    usr = {'tot_pings': tot_pings}
    if home.shape[0] == 1:
        home = home.iloc[0]
        usr['home_' + loc_col] = int(home[loc_col])
        usr[latColName + '_home'] = home[lat_col]
        usr[lonColName + '_home'] = home[lon_col]
        usr['home_pings'] = home['home_hour_count']
    elif home.shape[0] == 0:
        usr['home_' + loc_col] = None
        usr[latColName + '_home'] = None
        usr[lonColName + '_home'] = None
        usr['home_pings'] = None
    else:
        raise RuntimeError('Found more than one home locs for user %r' % g[uidColName].iloc[0])
        
    if work.shape[0] == 1:
        work = work.iloc[0]
        usr['work_' + loc_col] = int(work[loc_col])
        usr[latColName + '_work'] = work[lat_col]
        usr[lonColName + '_work'] = work[lon_col]
        usr['work_pings'] = work['work_hour_count']
    elif work.shape[0] == 0:
        usr['work_' + loc_col] = None
        usr[latColName + '_work'] = None
        usr[lonColName + '_work'] = None
        usr['work_pings'] = None
    else:
        raise RuntimeError('Found more than one work locs for user %r' % g[uidColName].iloc[0])
        
        
    return pd.Series(usr)


def _cycle_dur_count_delta(
                       min_durations=[0],
                       min_day_counts=[0],
                       min_hour_counts=[0],
                       min_delta_counts=[0],
                       min_delta_durations=[0]):
    '''
    Generator.
    Nestedly cycles through the args in their order and yields a numpy array
    with their values.
    '''
    for min_dur in min_durations:
        for cnt_day in min_day_counts:
            for cnt_hour in min_hour_counts:
                for delta_cnt in min_delta_counts:
                    for delta_dur in min_delta_durations:
                        yield np.array([
                            min_dur,
                            cnt_day,
                            cnt_hour,
                            delta_cnt,
                            delta_dur,
                        ])


def _keep_flag_hw_user(g,
                       min_durations=[0],
                       min_day_counts=[0],
                       min_hour_counts=[0],
                       min_delta_counts=[0],
                       min_delta_durations=[0],
                       limit_hw_locs=False,
                       loc_col=locColName,
                      ):
    '''
    Given a user groups of locations (tiles) with home work stats as returned by
    :attr:`mobilkit.stats.stopsToHomeWorkStats` it computes the home and work presence
    at different thresholds of home and work duration count etc.
    
    Parameters
    ----------
    g : group
        The grouped locations (or tiles) stats of the user.
    min_durations : iterable, optional
        The minimum duration of home and work stops to keep lines in the group.
    min_day_counts, min_hour_counts : iterable, optional
        The minimum count of stops in home and work locations to keep lines
        in the group.
    min_delta_counts, min_delta_durations : iterable, optional
        The minimum fraction of home/work hours during which the area/location is the
        most visited in terms of duration/count of stops for it to be kept.
    limit_hw_locs : bool, optional
        If `True`, it will limit the home and work candidates to the row(s) featuring
        `isHome` or `isWork` equal `True`, respectively. If `False` (default), all the
        rows are kept as candidates.
    loc_col : str, optional
        The column to use to check if the home and work candidates are in the same
        location.
    
    Returns
    -------
    user_flags : pd.Series
        A series containing, for each combination of threshold values in the order of
        minimum duration, minimum days, minimum hours, min delta count, min delta
        duration, the flag of:
        - `out_flags` if the user has a home AND work candidate with the threshold;
        - `out_has_home` if the user has a home candidate with the threshold;
        - `out_has_work` if the user has a work candidate with the threshold;
        - `out_same_locs` if the user has a unique the home and work candidate falling
          under the same `loc_col` ID.
    '''
    n_ele = len(min_durations)*len(min_day_counts)*len(min_hour_counts)\
                *len(min_delta_counts)*len(min_delta_durations)
    
    cols_values_home = ['home_duration', 'home_day_count', 'home_hour_count',
                   'home_delta_duration', 'home_delta_count']
    cols_values_work = ['work_duration', 'work_day_count', 'work_hour_count',
                   'work_delta_duration', 'work_delta_count']
    if limit_hw_locs:
        if not (np.any(g['isHome']) and np.any(g['isWork'])):
            return pd.Series({
                'out_flags': [False]*n_ele,
                'out_has_home': [False]*n_ele,
                'out_has_work': [False]*n_ele,
                'out_same_locs': [False]*n_ele,
            })
        tmp_rows = g.query('isHome==True')
        home_ref_values = tmp_rows[cols_values_home].values
        home_locs_values = tmp_rows[loc_col].values
        
        tmp_rows = g.query('isWork==True')
        work_ref_values = tmp_rows[cols_values_work].values
        work_locs_values = tmp_rows[loc_col].values
    else:
        home_ref_values = g[cols_values_home].values
        work_ref_values = g[cols_values_work].values
        
        home_locs_values = g[loc_col].values
        work_locs_values = g[loc_col].values
        
    
    out_has_home = []
    out_has_work = []
    out_flags = []
    out_same_loc = []
    for tmp_thres in _cycle_dur_count_delta(
                       min_durations=min_durations,
                       min_day_counts=min_day_counts,
                       min_hour_counts=min_hour_counts,
                       min_delta_counts=min_delta_counts,
                       min_delta_durations=min_delta_durations):
        tmp_thres = tmp_thres.reshape(1,-1)
        # Check for home
        home_mask = np.all(home_ref_values >= tmp_thres, axis=1)
        has_home = np.any(home_mask)
        out_has_home.append(has_home)
        # Check for work
        work_mask = np.all(work_ref_values >= tmp_thres, axis=1)
        has_work = np.any(work_mask)
        out_has_work.append(has_work)
        
        out_flags.append(has_home and has_work)
        if has_home and has_work:
            candidates_home = home_locs_values[home_mask]
            candidates_work = work_locs_values[work_mask]
            out_same_loc.append(len(candidates_home) == 1
                                and set(candidates_home)==set(candidates_work))
        else:
            out_same_loc.append(False)
        
    assert len(out_flags) == n_ele
    assert len(out_has_home) == n_ele
    assert len(out_has_work) == n_ele
    assert len(out_same_loc) == n_ele
    return pd.Series({
                'out_flags': out_flags,
                'out_has_home': out_has_home,
                'out_has_work': out_has_work,
                'out_same_locs': out_same_loc,
    })