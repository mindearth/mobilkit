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
from dask import dataframe as dd
from dask import array as da
import dask.bag as db

from mobilkit.dask_schemas import nunique

import numpy as np
import pandas as pd
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


def userHomeWorkLocation(df_hw):
    '''Given a dataframe returned by :attr:`mobilkit.stats.userHomeWork` computes, for each user, the home and work area as well as their location.
    The home/work area is the one with more pings recorded and the location is assigned to the mean point of this cloud.

    Parameters
    ----------

    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at least the `uid`, `tile_ID` and `isHome` and `isWork` columns.

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

    df_hw_cnt = df_hw.groupby([uidColName, zidColName])\
                    [["isHome","isWork"]].agg("sum").reset_index()
    df_hw_max = df_hw_cnt.groupby(uidColName)[["isHome","isWork"]].agg("max")
    
    df_hw_tot = df_hw_cnt.set_index(uidColName)\
                            .join(df_hw_max, rsuffix="_max")\
                            .reset_index()
    
    df_h_tot = df_hw_tot[df_hw_tot["isHome"] == df_hw_tot["isHome_max"]]\
                    .drop_duplicates(uidColName)
    df_w_tot = df_hw_tot[df_hw_tot["isWork"] == df_hw_tot["isWork_max"]]\
                    .drop_duplicates(uidColName)
    
    loc_home = df_hw.merge(df_h_tot[[uidColName,zidColName,"isHome_max"]],
                           on=[uidColName,zidColName], how="inner")\
                        [[uidColName,zidColName,latColName,lonColName,"isHome_max"]]\
                        .rename(columns={"isHome_max": "pings"})\
                        .reset_index().groupby(uidColName).agg({
                                "pings": "first",
                                zidColName: "first",
                                latColName: "mean",
                                lonColName: "mean",
                            })
    
    loc_work = df_hw.merge(df_w_tot[[uidColName,zidColName,"isWork_max"]],
                           on=[uidColName,zidColName], how="inner")\
                        [[uidColName,zidColName,latColName,lonColName,"isWork_max"]]\
                        .rename(columns={"isWork_max": "pings"})\
                        .reset_index().groupby(uidColName).agg({
                                "pings": "first",
                                zidColName: "first",
                                latColName: "mean",
                                lonColName: "mean",
                            })
    
    df_hw_loc = loc_home.join(loc_work, uidColName,
                              how="outer", lsuffix="_home", rsuffix="_work").reset_index()
    df_hw_loc = df_hw_loc.rename(columns={
                                    "tile_ID_home": "home_tile_ID",
                                    "tile_ID_work": "work_tile_ID",
                                    "pings_work": "work_pings",
                                    "pings_home": "home_pings",
                                    })

    return df_hw_loc


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

