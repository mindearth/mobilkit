# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

'''Tools and functions to analyze the data in time.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from datetime import datetime, timedelta

from dask import dataframe as dd
from dask import array as da
import dask.bag as db
from mobilkit.dask_schemas import nunique, unique

from haversine import haversine

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


from copy import copy, deepcopy

from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO
# from mobilkit.tools import flattenSetsUdf

# from spark_types import eventLineDTzone


def filter_daynight_time(df,
                         filter_from_h=21.5,
                         filter_to_h=8.5,
                         previous_day_until_h=4.,
                         daytime_from_h=9.0,
                         daytime_to_h=21.0,
                        ):
    '''
    Prepares a raw event df for the ping-based displacement analysis.
    
    Parameters
    ----------
    df : dask.DataFrame
        A dataframe containing at least the `uid,datetime,lat,lng` columns as
        returned by :attr:`mobilkit.loader.load_raw_files` or similar functions.
    filter_{from,to}_h : float
        The starting and ending float hours to consider.
        If `from_hour<to_hour` only pings whose float hour `h` are
        `from_hour <= h < to_hour` are considered otherwise all the pings with
        `h >= from_hour` or `h < to_hour`.
        Note that float hour `h` for datetime `dt` is `h = dt.hour + dt.minute/60.`
        so to express 9:45am put `9.75`.
    previous_day_until_h : float
        All the valid events with float hour `h < previous_day_until_h` will be 
        projected to the previous day. Put 0 or a negative number to keep all events
        of one day to its `date`.
    daytime_{from,to}_h : float
        The starting and ending float hours to consider in daytime (other will be put
        in nightime. All events with `from_hour<= float_hour <= to_hour` will have a
        1 entry in the daytime column, others 0. from hour **must** be smaller than
        to hour.
        Note that float hour `h` for datetime `dt` is `h = dt.hour + dt.minute/60.`
        so to express 9:45am put `9.75`.
    
    Returns
    -------
    df : dask.DataFrame
        The same initial dataframe filtered accordingly to `from_hour,to_hour` and
        with three additional columns:
        
        - `float_hour`: the day-hour expressed as `h=dt.hour + dt.minutes`
        - `date`: the `datetime` column floored to the day. All events with
            `float_hour < previous_day_until_h` will be further advanced by one
            day.
        - `daytime`: 1 if the event's `float_hour` is between `daytime_from_h` and
            `daytime_to_h`
    '''
    assert daytime_from_h < daytime_to_h
    
    df_with_hour = df.assign(float_hour=df[dttColName].dt.hour
                                        + df[dttColName].dt.minute / 60.)
    if filter_from_h > filter_to_h:
        df_filtered = df_with_hour[
                            (df_with_hour["float_hour"] >= filter_from_h)
                            | (df_with_hour["float_hour"] < filter_to_h)
                        ]
    else:
        df_filtered = df_with_hour[
                            (df_with_hour["float_hour"] >= filter_from_h)
                            & (df_with_hour["float_hour"] < filter_to_h)
                        ]
    df_withDay = df_filtered.assign(
                    date=df_filtered[dttColName].dt.floor("1D"),
                    daytime=df_filtered["float_hour"].between(daytime_from_h, daytime_to_h))
    df_fixed = df_withDay.assign(date=df_withDay["date"]
                                 - dd.to_timedelta((df_withDay["float_hour"]
                                            < previous_day_until_h).astype(int),
                                        unit="d"))
    return df_fixed




def computeTimeBinActivity(df, byArea=False, timeBin="hour", split_out=10):
    '''Basic function to compute, for each time bin and area, the activity profile in terms of
    users and pings recorded. It also computes the set of users seen in that bin for later aggregations.

    Parameters
    ----------
    df : dask.DataFrame
        A dataframe as returned from :attr:`mobilkit.loader.load_raw_files` or imported from
        ``scikit-mobility`` using :attr:`mobilkit.loader.load_from_skmob`. If using ``byArea``
        the df must contain the ``tile_ID`` column as returned by :attr:`mobilkit.spatial.tessellate`.
    byArea : bool, optional
        Whether or not to compute the activity per area (default ``False``).
        If ``False`` will compute the overall activity.
    timeBin : str, optional
        The width of the time bin to use to aggregate activity. Must be one of the ones
        found in [pandas time series aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases). For instance:
            - `'B'` business day frequency
            - `'D'` calendar day frequency
            - `'W'` weekly frequency
            - `'M'` month end frequency
            - `'MS'` month start frequency
            - `'SMS'` semi-month start frequency (1st and 15th)
            - `'BH'` business hour frequency
            - `'H'` hourly frequency
            - `'T','min'` minutely frequency
    split_out : int, optional
        The number of dask dataframe partitions after the groupby aggregation.
            
    Returns
    -------
    df_activity : dask.dataframe
        A dataframe with these columns:

        - one with the same name as ``timeBin`` with the date truncated at the selected width.

        - ``pings`` the number of pings recorded in that time bin and area (if ``byArea=True``).

        - ``users`` the number of users seen in that time bin and area (if ``byArea=True``).

        - ``users_set`` the set of users seen in that time bin and area (if ``byArea=True``). Useful to normalize later analysis.
        - ``pings_per_user`` the average number of pings per user in that time bin and area (if ``byArea=True``).
        - ``tile_ID`` (if ``byArea=True``) the area where the signal has been recorded.
    '''
    
    aggKeys = [timeBin]
    if byArea:
        aggKeys.append(zidColName)
    df_out = copy(df)
    df_out[timeBin] = df_out[dttColName].dt.round(timeBin)
    df_out = df_out.groupby(aggKeys)[[uidColName]]\
                            .agg(["count", nunique, unique], split_out=split_out)
    
    # Flatten columns and rename
    df_out.columns = ['_'.join(col).strip() for col in df_out.columns.values]
    df_out = df_out.rename(columns={
                                uidColName + "_count": "pings",
                                uidColName + "_nunique": "users",
                                uidColName + "_unique": "users_set",})
    df_out["pings_per_user"] = df_out["pings"]  / df_out["users"].clip(lower=1.)
    return df_out


def plotMonthlyActivity(df_activity, timeBin, what="users", ax=None, log_y=False, **kwargs):
    '''Basic function to plot the monthly activity of areas or total region.

    Parameters
    ----------
    df_activity : dask.DataFrame
        A dataframe as returned from :attr:`mobilkit.temporal.computeTimeBinActivity`.
    timeBin : str
        The width of the time bin used in :attr:`mobilkit.temporal.computeTimeBinActivity`.
    what : str, optional
        The quantity to plot. Must be one amongst ``'users', 'pings', 'pings_per_user'``.
    ax : axis, optional
        The axis to use. If ``None`` will create a new figure.
    log_y : bool, optional
        Whether or not to plot with y log scale. Default ``False``.
    **kwargs
        Will be passed to ``seaborn.lineplot`` function.

    Returns
    -------
    df : pandas.DataFrame
            Thee aggregated data plotted.
    ax : axis
        The axis of the figure.
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(18,6))


    df = copy(df_activity)
    df["month"] = df[timeBin].dt.round("MS")
    df["month_hour"] = df[timeBin].dt.dayofmonth()*24 + df["timeBin"].dt.hour
    df = df.compute()
#     df = df_activity.withColumn("month", sqlF.date_trunc("month", sqlF.col(timeBin)))\
#                     .withColumn("month_hour", sqlF.dayofmonth(timeBin)*24 + sqlF.hour(timeBin))

    sns.lineplot("month_hour", what, hue="month", data=df, ax=ax, **kwargs)
    if log_y: ax.set_yscale("log")
    hours_per_day = 24
    locs = np.arange(hours_per_day,hours_per_day*32,hours_per_day)
    plt.xticks(locs, ["%d"%(l//hours_per_day) for l in locs], size=14)
    plt.yticks(size=14)

    plt.xlabel("Day", size=16)
    plt.ylabel("Users" if what == "users"
            else "Pings" if what == "pings"
            else "Pings per user", size=16)

    plt.legend(fontsize=14, bbox_to_anchor=(1.01,.5), loc="center left")
    plt.tight_layout()

    return df, ax


def computeTemporalProfile(df_tot, timeBin,
                           byArea=False,
                           profile="week",
                           weekdays=None,
                           normalization=None,
                           start_date=None,
                           stop_date=None,
                           date_format=None,
                           sliceName=None,
                           selected_areas=None,
                           areasName=None):
    '''Function to compute the normalized profiles of areas.
    The idea is to have a dataframe with the count of users and pings
    per time bin (and per area is ``byArea=True``) together with a
    normalization column (computed if ``normalization`` is not ``None``
    over a different time window ``profile``) telling the total number
    of pings and users seen in that period (and in that area if
    ``byArea``).
    If ``normalization`` is specified, also the fraction of users and
    pings recorded in an area at that time bin are given.

    Parameters
    ----------
    df_tot : dask.DataFrame
        A dataframe as returned from :attr:`mobilkit.loader.load_raw_files`
        or imported from ``scikit-mobility`` using :attr:`mobilkit.loader.load_from_skmob`.
        If using ``byArea`` the df must contain the ``tile_ID`` column
        as returned by :attr:`mobilkit.spatial.tessellate`.
    timeBin : str
        The width of the time bin to use to aggregate activity.
        Currently supported: ["W", "MS", "M", "H", "D", "T"]
        You can implement others found in [pandas time series aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases). For instance:
            - `'B'` business day frequency
            - `'D'` calendar day frequency
            - `'W'` weekly frequency
            - `'M'` month end frequency
            - `'MS'` month start frequency
            - `'SMS'` semi-month start frequency (1st and 15th)
            - `'BH'` business hour frequency
            - `'H'` hourly frequency
            - `'T','min'` minutely frequency
    byArea : bool, optional
        Whether or not to compute the activity per area (default ``False``).
        If ``False`` will compute the overall activity.
    profile : str
        The base of the activity profile: must be ``"week"`` to compute the
        weekly profile or ``"day"`` for the daily one or ``"month"`` for
        one month period (`month_end` to use month end).
        Each profile of area / week or day (depending on profile) will be
        computed separately.
        **NOTE** that this period should be equal or longer than the ``timeBin``
        (i.e., ``"weekly"`` or ``"monthly"`` if ``timeBin="week"``) otherwise
        the normalization will fail.
    weekdays : set or list, optional
        The weekdays to consider (1 Sunday -> 7 Saturday).
        Default ``None`` equals to keep all.
    normalize : str, optional
        One of ``None, "area", "total"``.
        Normalize nothing (``None``), on the total period of the area
        (``area``) or on the total period of all the selected areas (``total``).
    start_date : str, optional
        The starting date when to consider data in the ``date_format`` format.
    stop_date : str, optional
        The end date when to consider. Must have the same format as ``start_date``.
    date_format : str, optional
        The python date format of the dates, if given.
    sliceName : str, optional
        The name that will be saved in `timeSlice` column, if given.
    selected_areas : set or list, optional
        The set or list of selected areas. If ``None`` (default) uses all the areas.
        Use :attr:`mobilkit.spatial.selecteAreasFromBounds` to select areas from given bounds.
    areasName : str, optional
        The name that will be saved in `areaName` column, if given.

    Returns
    -------
    df_normalized : dask.DataFrame
        A dataframe with these columns:
        - one with the same name as ``timeBin`` with the date truncated at the selected width.
        
        - ``pings`` the number of pings recorded in that time bin and area (if ``byArea=True``).

        - ``users`` the number of users seen in that time bin and area (if ``byArea=True``).

        - ``pings_per_user`` the average number of pings per user in that time bin and area
                (if ``byArea=True``).
        - ``tile_ID`` (if ``byArea=True``) the area where the signal has been recorded.
        
        - the additional columns ``timeSlice`` and ``areaName``if the two names are given,
                plus, if ``normalization`` is not ``None``:

        - ``tot_pings/users`` the total number of pings and users seen in the area (region) in
                the profile period if normalize is ``"area"`` (``total``).

        - ``frac_pings/users`` the fraction of pings and users seen in that area, at that time bin
                with respect to the total volume of the area (region) depending on the normalization.

        - ``profile_hour`` the zero-based hour of the typical month, week or day (depending on the
                value of ``profile``).
    '''

    assert normalization in [None, "area", "total"]
    assert timeBin in ["W","MS","M","H","D","T","min"]
    if normalization == "area":
        assert byArea == True
    profile_dict = {"week": "W", "month": "MS", "day": "D", "month_end": "M"}
    assert profile in profile_dict
    profile_pandas = profile_dict[profile]

    df = copy(df_tot)
    
    if start_date is not None:
        df = df[df[dttColName] >= datetime.strptime(start_date, date_format)]
    if stop_date is not None:
        df = df[df[dttColName] < datetime.strptime(stop_date, date_format)]
    if selected_areas is not None:
        df = df[df[zidColName].isin(list(selected_areas))]

    if weekdays is not None:
        weekdays = set([int(i) for i in weekdays])
        df["ddooww"] = df[dttColName].dt.weekday
        df = df[df["ddooww"].isin(weekdays)]
        df = df.drop('ddooww', axis=1)

    aggKeys = [timeBin]
    if byArea:
        aggKeys.append(zidColName)

    # Compute number of users and pins per time bin
    if timeBin == "W":
        df[timeBin] = (df[dttColName] - dd.to_timedelta(df[dttColName].dt.weekday, unit='d')).dt.floor("D")
    elif timeBin == "MS":
        df[timeBin] = (df[dttColName] - dd.to_timedelta(df[dttColName].dt.day - 1, unit='d')).dt.floor("D")
    elif timeBin == "M":
        df[timeBin] = (df[dttColName] + dd.to_timedelta(df[dttColName].dt.days_in_month
                                                        -df[dttColName].dt.day,
                                                        unit='d')).dt.floor("D")
    else:
        df[timeBin] = df[dttColName].dt.floor(timeBin)
        
    # Add the period
    if profile_pandas == "MS":
        df[profile] = (df[timeBin]
                           - dd.to_timedelta(df[timeBin].dt.day - 1,
                                             unit='d')).dt.floor("D")
    elif profile_pandas == "M":
        df[profile] = (df[timeBin] +
                       dd.to_timedelta(df[timeBin].dt.days_in_month
                                       -df[timeBin].dt.day,
                                    unit='d')).dt.floor("D")
    elif profile_pandas == "W":
        df[profile] = (df[timeBin]
                       - dd.to_timedelta(df[timeBin].dt.weekday, unit='d')).dt.floor("D")
    else:
        df[profile] = df[timeBin].dt.floor(profile_pandas)

#     df_reduced = df.groupby(aggKeys).agg(
#                         {uidColName: ["count", nunique],
#                             profile: ["first"],})
#     # Flatten columns and rename
#     df_reduced.columns = ['_'.join(col).strip() for col in df_reduced.columns.values]
#     df_reduced = df_reduced.rename(columns={
#                                 uidColName + "_count": "pings",
#                                 uidColName + "_nunique": "users",
#                                 profile + "_first": profile,})
    
#     df_reduced["pings_per_user"] = df_reduced["pings"] / df_reduced["users"].clip(lower=1.)
#     df_reduced = df_reduced.reset_index()
    if normalization is None:
        return_meta = dict(**df.dtypes)
        return_meta = {k: return_meta[k] for k in [timeBin,profile,zidColName]}
        return_meta["pings"] = np.int64
        return_meta["users"] = np.int64
        return_meta["pings_per_user"] = np.float64
        if byArea:
            # print("No norm, area", df)
            df_reduced = df_reduced.assign(col1=df_reduced[zidColName])
            df_reduced = df.groupby(col1)\
                                    .apply(_computePerAreaGrouped,
                                           binKey=timeBin, profile=profile,
                                           meta=return_meta)
        else:
            df_reduced = _computePerAreaGrouped(df,
                                                binKey=timeBin,
                                                profile=profile)
    else:
        return_meta = dict(**df.dtypes)
        return_meta = {k: return_meta[k] for k in [timeBin, zidColName, profile]}
        return_meta["pings"] = np.int64
        return_meta["users"] = np.int64
        return_meta["pings_per_user"] = np.float64
        return_meta["tot_pings"] = np.int64
        return_meta["tot_users"] = np.int64
        return_meta["frac_pings"] = np.float64
        return_meta["frac_users"] = np.float64
        
        if byArea:
            df = df.assign(col1=df[zidColName], col2=df[profile])
            aggNorm = ["col1","col2"]
            # levelDrop = 2
        else:
            df = df.assign(col1=df[profile])
            aggNorm = ["col1"]
            # aggNorm = [profile]
            # levelDrop = 1
        # print(df.head(2))
        # print(aggNorm, levelDrop, timeBin, return_meta)
        df_reduced = df.groupby(aggNorm)\
                        .apply(_computePerAreaGroupedNormalization,
                               meta=return_meta, binKey=timeBin, profileKey=profile)
#         try:
#             # print(df_reduced.head(2))
#             if levelDrop > 1:
#                 df_reduced = df_reduced.map_partitions(my_droplevel,
#                                                    level=levelDrop,
#                                                    meta=return_meta)
#             else:
#                 #TODO not working on dask, don't know why...
#                 # df_reduced = df_reduced.reset_index()
#                 # df_reduced = df_reduced.drop("level_1", axis=1)
#                 pass
#             df_reduced = df_reduced.reset_index()
#         except Exception as E:
#             print("Warning, got exc:", str(E))
            
    df_reduced["profile_hour"] = df_reduced[timeBin].dt.hour
    if profile == "week":
        df_reduced["profile_hour"] = df_reduced["profile_hour"] +\
                                        df_reduced[timeBin].dt.weekday*24
    elif profile in ["month","month_end"]:
        df_reduced["profile_hour"] = df_reduced["profile_hour"] +\
                                        (df_reduced[timeBin].dt.day - 1) *24
    if sliceName is not None:
        assert type(sliceName) is str
        df_reduced = df_reduced.assign(timeSlice=sliceName)
    if areasName is not None:
        assert type(areasName) is str
        df_reduced = df_reduced.assign(areaName=areasName)

    return df_reduced


def my_droplevel(df, level=1):
    df.index = df.index.droplevel(level)
    return df


def _computePerAreaGroupedNormalization(g, binKey=None, profileKey=None):
    # For each profile, area compute totals and per bins totals
    # Totals
    if type(g) == dd.core.DataFrame:
        df_reduced = g.groupby(binKey)\
                        .agg({latColName: "count",
                              uidColName: nunique,
                              zidColName: "first",
                             profileKey: "first"})
    else:
        df_reduced = g.groupby(binKey)\
                        .agg({latColName: "count",
                              uidColName: "nunique",
                             zidColName: "first",
                             profileKey: "first"})
        
    df_reduced = df_reduced.rename(columns={latColName: "pings", uidColName: "users"})
    df_reduced = df_reduced.assign(
            pings_per_user=df_reduced["pings"]/df_reduced["users"],
            tot_pings=df_reduced["pings"].sum(),
            tot_users=df_reduced["users"].sum(),
        )
    df_reduced = df_reduced.assign(
            frac_pings=df_reduced["pings"] / df_reduced["tot_pings"],
            frac_users=df_reduced["users"] / df_reduced["tot_users"],
        )
    df_reduced = df_reduced.reset_index()
    
    return df_reduced[[binKey, zidColName, profileKey, "pings","users","pings_per_user",
                      "tot_pings","tot_users","frac_pings","frac_users"]]


def _computePerAreaGrouped(g, binKey=None, profile=None):
    if type(g) == dd.core.DataFrame:
        df_reduced = g.groupby(binKey).agg({uidColName: ["count", nunique],
                                            profile: ["first"],
                                            zidColName: ["first"]}).reset_index()
    else:
        df_reduced = g.groupby(binKey).agg({uidColName: ["count", "nunique"],
                                            profile: ["first"],
                                            zidColName: ["first"]}).reset_index()
    # Flatten columns and rename
    df_reduced.columns = ['_'.join(col).strip() if len(col[-1])>0 else col[0]
                                  for col in df_reduced.columns.values]
    df_reduced = df_reduced.rename(columns={
                                uidColName + "_count": "pings",
                                uidColName + "_nunique": "users",
                                profile + "_first": profile,
                                zidColName + "_first": zidColName,
                            })
    df_reduced["pings_per_user"] = df_reduced["pings"] / df_reduced["users"].clip(lower=1.)
    df_reduced = df_reduced[[binKey, profile, zidColName, 'pings', 'users', 'pings_per_user']]
    # df_reduced = df_reduced.reset_index()
    return df_reduced    

def computeResiduals(df_activity, signal_column, profile):
    '''Function that computes the average, z-score and residual activity of an area in a given time
    period and for a given time bin.

    Parameters
    ----------
    df_activity : dask.DataFrame
        As returned by :attr:`mobilkit.temporal.computeTemporalProfile`, a dataframe with the columns
        and periods volumes and normalization (if needed) already computed.
    profile : str
        The temporal profile used for normalization in :attr:`mobilkit.temporal.computeTemporalProfile`.
    signal_column : str
        The columns to use as proxy for volume. Usually one of ``"users", "pings", "frac_users", "frac_pings"``

    Returns
    -------
    results, mappings
        Two dictionaries containing the aggregated results in numpy arrays.
        ``results`` has four keys:

            - ``raw`` the raw signal in the ``area_index,period_index,period_hour_index`` indexing;

            - ``mean`` the mean over the periods of the raw signal in the
                    ``area_index,period_hour_index`` shape;

            - ``zscore`` the zscore of the area signal (with respect to its average and std) in the
                    ``area_index,period_hour_index`` shape;

        - ``residual`` the residual activity computed as the difference between the area's ``zscore``
                    and the global average ``zscore`` at a given hour in the ``area_index,period_hour_index``
                    shape;

        On the other hand, ``mappings`` contains the back and forth mapping between the numpy indexes
        and the original values of the areas (``idx2area`` and ``area2idx``), periods, and, hour of the period.
        These will be useful later for plotting.
    '''
#     uniques = df_activity.select(
#                 sqlF.collect_set(zidColName).alias("areas"),
#                 sqlF.collect_set(profile).alias("periods"),
#                 sqlF.collect_set("profile_hour").alias("hours"),
#             ).toPandas()
    set_areas = set([d for d in df_activity[zidColName].unique()])# .compute())# set(uniques.loc[0,"areas"])
    set_periods = set([pd.to_datetime(d) for d in df_activity[profile].unique()])# .compute())# set(uniques.loc[0,"areas"])
    set_hours = set([d for d in df_activity["profile_hour"].unique()])# .compute())# set(uniques.loc[0,"areas"])
    # set_periods = set(uniques.loc[0,"periods"])
    # set_hours = set(uniques.loc[0,"hours"])

    # Compute the mappings
    area2idx = {k: v for v, k in enumerate(sorted(set_areas))}
    week2idx = {k: v for v, k in enumerate(sorted(set_periods))}
    hour2idx = {k: v for v, k in enumerate(sorted(set_hours))}
    idx2area = dict(map(reversed,area2idx.items()))
    idx2week = dict(map(reversed,week2idx.items()))
    idx2hour = dict(map(reversed,hour2idx.items()))
    nAreas = len(area2idx)
    nWeeks = len(week2idx)
    nHours = len(hour2idx)
    
    zone_hour_volume = np.zeros((nAreas,nWeeks,nHours))
    tmp_df = df_activity[[zidColName,profile,"profile_hour",signal_column]]# .compute()
    for tid, prof, prof_h, val in tmp_df[
            [zidColName, profile, "profile_hour", signal_column]].values:
        try:
            i = area2idx[tid]
        except KeyError:
            continue
        j = week2idx[prof]
        k = hour2idx[prof_h]
        zone_hour_volume[i,j,k] = val

    avg_zone_hour_volume = np.mean(zone_hour_volume, axis=1)

    zsc_zone_hour_volume = (avg_zone_hour_volume -
                                    avg_zone_hour_volume.mean(axis=-1,keepdims=True))
    tmp_std = avg_zone_hour_volume.std(axis=-1,keepdims=True)
    tmp_std = np.where(tmp_std==0, np.ones_like(tmp_std), tmp_std)
    tmp_std = np.where(np.isnan(tmp_std), np.ones_like(tmp_std), tmp_std)
    zsc_zone_hour_volume /= tmp_std

    res_zone_hour_volume = zsc_zone_hour_volume - zsc_zone_hour_volume.mean(axis=0, keepdims=True)

    results = {
        "raw": zone_hour_volume,
        "mean": avg_zone_hour_volume,
        "zscore": zsc_zone_hour_volume,
        "residual": res_zone_hour_volume,
    }

    mappings = {
        "area2idx": area2idx,
        "hour2idx": hour2idx,
        "period2idx": week2idx,
        "idx2area": idx2area,
        "idx2hour": idx2hour,
        "idx2period": idx2week,
    }

    return results, mappings


def homeLocationWindow(df_hw,
                       initial_days_home=None,
                       home_days_window=3,
                       start_date=None,
                       stop_date=None):
    '''
    Given a dataframe returned by :attr:`mobilkit.stats.userHomeWork` computes,
    for each user, the home area for every window of ``home_days_window`` days
    after the initial date.
    Note that the points before 12pm will be assigned to the previous day's night
    and the one after 12pm to the same day's night.

    Parameters
    ----------
    df_hw : dask.dataframe
        A dataframe as returned by :attr:`mobilkit.stats.userHomeWork` with at
        least the `uid`, `tile_ID`, `datetime` and `isHome` and `isWork` columns.
    initial_days_home : int, optional
        The number of initial days to be used to compute the original home area.
        If ``None`` (default) it will just compute the home for every window
        since the beginning.
    home_days_window : int, optional
        The number of days to use to assess the home location of a user (default 3).
        For each day ``d`` in the ``start_date`` to ``stop_date - home_days_window``
        it computes the home location between the ``[d,d+home_days_window)`` period.
    start_date : datetime.datetime
        A python datetime object with no timezone telling the date (included) to
        start from. The default behavior is to keep all the events.
    stop_date : datetime.datetime, optional
        A python datetime object with no timezone telling the date (excluded) to
        stop at. Default is to keep all the events.

    Returns
    -------
    df_hwindow : pandas.dataframe
        The dataframe containing, for each user and active day of user the
        ``tile_ID`` of the user's home and the number of pings recorded there in
        the time window. The date is saved in ``window_date`` and refers to the
        start of the time window (whose index is saved in ``timeSlice``).
        For the initial home window the date corresponds to its end.
        
    Note
    ----
    When determining the home location of a user, please consider that some data providers, like _Cuebiq_, obfuscate/obscure/alter the coordinates of the points falling near the user's home location in order to preserve privacy.

    This means that you cannot locate the precise home of a user with a spatial resolution higher than the one used to obfuscate these data. If you are interested in the census area (or geohash) of the user's home alone and you are using a spatial tessellation with a spatial resolution wider than or equal to the one used to obfuscate the data, then this is of no concern.

    However, tasks such as stop-detection or POI visit rate computation may be affected by the noise added to data in the user's home location area. Please check if your data has such noise added and choose the spatial tessellation according to your use case.
    '''
    if initial_days_home is not None:
        assert initial_days_home > 0
    assert home_days_window > 0 and type(home_days_window) is int

    # Prepare the column with the day to which a row is assigned
    # and its distance to initial date
    filtered_df = df_hw[[uidColName,zidColName,dttColName,"isHome"]]
    filtered_df = filtered_df[filtered_df["isHome"] == 1]
    filtered_df = filtered_df.assign(
                        hour=filtered_df[dttColName].dt.hour,
                        day=filtered_df[dttColName].dt.floor("D"))
    
    filtered_df = filtered_df.assign(sday=(filtered_df["hour"] < 12).astype(int))
    filtered_df["day"]  = filtered_df["day"]\
                                - dd.to_timedelta(filtered_df["sday"], unit='d')
    filtered_df = filtered_df[[uidColName,zidColName,dttColName,"isHome","day"]]

    if start_date is not None:
        filtered_df = filtered_df[filtered_df["day"] >= start_date]
    else:
        start_date = filtered_df["day"].min().compute()
    if stop_date is not None:
        filtered_df = filtered_df[filtered_df["day"] < stop_date]
        
    # Compute once the number of pings per zone per day per user
    filtered_df = filtered_df.assign(level0=filtered_df[uidColName],
                                     level1=filtered_df[zidColName],
                                     level2 =filtered_df["day"]
                                    ).groupby(["level0","level1","level2"])\
                                    .agg({uidColName: "first",
                                          zidColName: "first",
                                          "day": "first",
                                          "isHome": "sum",
                                         }).rename(columns={"isHome": "pings"})

    filtered_df = filtered_df.assign(day0=start_date)
    filtered_df = filtered_df.assign(deltaDay=(filtered_df["day"]
                                               - filtered_df["day0"]).dt.days)
    filtered_df = filtered_df.map_partitions(lambda p: p.reset_index(drop=True))
    filtered_df = filtered_df.persist()
    
    print("Got the delta days distributed as:",
          filtered_df["deltaDay"].compute().describe())
    
    # First slice and set initial date for windows
    if initial_days_home is not None:
        initial_df = filtered_df[filtered_df["deltaDay"] < initial_days_home]
        first_date_of_windows = initial_df["day"].max().compute()
        initial_df = initial_df.assign(timeSlice=0,
                                       level0=initial_df[uidColName],
                                      )

        return_meta = dict(**initial_df.dtypes)
        return_meta = {k: return_meta[k] for k in [uidColName,zidColName,
                                                   "timeSlice"]}
        return_meta["pings"] = np.int64

        initial_df = initial_df.groupby("level0")\
                                .apply(_computeUserSliceWindows, meta=return_meta)\
                                .compute().reset_index(drop=True)
    else:
        initial_df = None
        initial_days_home = 0
        first_date_of_windows = filtered_df["day"].max().compute()

    # Now for each window I do the same and concat to the original one
    offset_windows = 1 if initial_days_home > 0 else 0
    for window_idx in range(home_days_window):
        tmp_day_0 = initial_days_home + window_idx
        slice_df = filtered_df[filtered_df["deltaDay"] >= tmp_day_0]

        slice_df = slice_df.assign(timeSlice=offset_windows + window_idx
                                        + home_days_window
                                               * ( (slice_df["deltaDay"] - tmp_day_0)
                                                          //home_days_window ))
        print("Doing window %02d / %02d" % (window_idx+1, home_days_window))

        slice_df = slice_df.assign(level0=slice_df["timeSlice"],
                                       level1=slice_df[uidColName],
                                      )
        return_meta = dict(**slice_df.dtypes)
        return_meta = {k: return_meta[k] for k in [uidColName,zidColName,
                                                   "timeSlice"]}
        return_meta["pings"] = np.int64

        slice_df = slice_df.groupby(["level0","level1"])\
                                .apply(_computeUserSliceWindows, meta=return_meta)\
                                .compute().reset_index(drop=True)


        if initial_df is None:
            initial_df = slice_df
        else:
            initial_df = pd.concat([initial_df, slice_df], sort=True, ignore_index=True)
            
    # Add reference date
    initial_df = initial_df.assign(window_date=first_date_of_windows
                                           +dd.to_timedelta(initial_df["timeSlice"],
                                                            unit="D"))
    return initial_df


def _computeUserSliceWindows(g):
    if type(g) == dd.core.DataFrame:
        g = g.assign(groupZID=g[zidColName]).groupby("groupZID").agg({
                            uidColName: "first",
                            zidColName: "first",
                            "timeSlice": "first",
                            "pings": "sum",
                        }).sort_values("pings", ascending=False)\
                        .reset_index(drop=True)
    else:
        g = g.groupby(zidColName).agg({
                            uidColName: "first",
                            "timeSlice": "first",
                            "pings": "sum",
                        }).sort_values("pings", ascending=False)\
                        .reset_index()
    
    return g[[uidColName,zidColName,"timeSlice","pings"]].iloc[0]


def computeDisplacementFigures(df_disp, minimum_pings_per_night=5):
    '''
    Given a dataframe returned by :attr:`mobilkit.temporal.homeLocationWindow` computes a pivoted
    dataframe with, for each user, the home area for every time window, plus the arrays of displaced
    and active people per area and the arrays with the (per user) cumulative number of  areas where
    the user slept.

    Parameters
    ----------
    df_disp : pandas.dataframe
        A dataframe as returned by :attr:`mobilkit.temporal.homeLocationWindow`.
    minimum_pings_per_night : int, optional
        The number of pings recorded during a night for a user to be considered.

    Returns
    -------
    df_pivoted, first_user_area, heaps_arrays, count_users_per_area : pandas.dataframe, dict, array, dict
        - ``df_pivoted`` is a dataframe containing one row per user and with the column being the sorted
            time windows of the analysis period. Each cell contains the location where the user (row)
            has slept in night t (column), ``Nan`` if the user was not active that night.
        
        - ``first_user_area`` is a dict telling, for each user, the ``tile_ID`` where he has been sleeping
            for the first time.
        
        - ``heaps_arrays`` is a (n_users x n_windows) array telling the cumulative number of areas where
            a users slept up to window t.
        
        - ``counts_users_per_area`` is a dictionary ``{tile_ID: {"active": [...], "displaced": [...]}}``
            telling the number of active and displaced people per area in time.
    '''
    
    init_df_joined_pd = df_disp.sort_values([uidColName,"timeSlice"])
    pivoted = init_df_joined_pd[
                    init_df_joined_pd["pings"]>=minimum_pings_per_night]\
                .pivot(uidColName, "window_date", zidColName)
    pivoted = pivoted[sorted(pivoted.columns)]
    
    areas_displacement = set([d for d in pivoted.values.flatten() if not np.isnan(d)])
    n_areas_displacement = len(areas_displacement)
    n_time_windows = pivoted.shape[1]
    
    pivoted_arra = pivoted.values
    prima_zona = np.zeros(pivoted_arra.shape[0])

    heaps = np.zeros_like(pivoted_arra)
    count_users_per_area = {a: {
                            "active": np.zeros(n_time_windows),
                            "displaced": np.zeros(n_time_windows),
                        } for a in areas_displacement}

    for i in range(pivoted_arra.shape[0]):
        row = pivoted_arra[i,:]
        tmp_set = set()
        assigned = False
        for j in range(pivoted_arra.shape[1]):
            e = row[j]
            if not np.isnan(e):
                tmp_set.add(e)
            if not assigned and len(tmp_set) == 1:
                prima_zona[i] = int(e)
                assigned = True

            if assigned and not np.isnan(e):
                tmp_zona_original = prima_zona[i]
                count_users_per_area[tmp_zona_original]["active"][j] += 1
                if e != tmp_zona_original:
                    count_users_per_area[tmp_zona_original]["displaced"][j] += 1
            heaps[i,j] = len(tmp_set)
    pivoted_arra.shape
    
    prima_zona = {u: prima_zona[i] for i, u in enumerate(pivoted.index)}
    
    return pivoted, prima_zona, heaps, count_users_per_area
    
    
    
def plotDisplacement(count_users_per_area, pivoted, gdf,
                     area_key="tile_ID",
                     epicenter=[18.584,98.399],
                     bins=5):
    '''
    Parameters
    ----------
    count_users_per_area : dict
        The dict returned with the pivot table, the original home location,
        and the Heaps law of visited areas by :attr:`mobilkit.temporal.homeLocationWindow`.
    pivoted : pandas.DataFrame
        The pivoted dataframe of the visited location during the night as returned with the
        the original home location, the Heaps law of visited areas and the count of users per
        area and date by :attr:`mobilkit.temporal.homeLocationWindow`.
    gdf : geopandas.GeoDataFrame
        The geodataframe used to tessellate data. Must contain the `area_key` column.
    area_key : str
        The column containing the ID of the tessellation areas used to join the displacement
        data and the GeoDataFrame.
    epicenter : tuple
        The `(lat,lon)` coordinates of the center to be used to split areas in `bins` bins
        based on their distance from this point.
    bins : int
        The number of linear distance bins to compute from the epicenter.
    
    Returns
    -------
    '''
    gdf = gdf.copy()
    dates_sorted = np.array(pivoted.columns)
    # Compute the distance of each area's centroid from epicenter
    gdf["distance_epic"] = gdf.geometry.centroid.apply(lambda p:
                                        haversine(epicenter,
                                                    (p.xy[1][0], p.xy[0][0]) ))
    
    # Bin areas depending on their distance
    distance_bins = np.linspace(0, max(gdf["distance_epic"])+1, bins+1)
    
    gdf["distance_bin"] = gdf["distance_epic"].apply(lambda v:
                                                     np.argmax(distance_bins>=v)-1)
    
    
    # For each bin's areas plot the displacement rate
    fig, ax = plt.subplots(1,1,figsize=(15,6))
    ymax = -1
    for dist_bin in range(len(distance_bins)):
        tmp_areas = set(gdf[gdf["distance_bin"]==dist_bin][area_key].values)
        tmp_areas = tmp_areas.intersection(count_users_per_area.keys())
        if len(tmp_areas) == 0:
            continue

        tmp_arra_disp = np.vstack([count_users_per_area[a]["displaced"] for a in tmp_areas])
        tmp_arra_act = np.vstack([count_users_per_area[a]["active"] for a in tmp_areas])
        tmp_Ys = tmp_arra_disp.sum(axis=0) / np.clip(tmp_arra_act.sum(axis=0), a_min=1., a_max=None)
        plt.plot(dates_sorted, tmp_Ys, label="Dist. bin %d"%dist_bin, lw=3)
        ymax = max(ymax, max(tmp_Ys))
    plt.vlines(dates_sorted[1], 0, ymax*2, lw=4, linestyles="--", color="r",
              label="First window")
    plt.ylim(0, ymax*1.1)

    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Fraction of active\n users displaced")
    plt.legend();
    
    return fig, gdf
    
