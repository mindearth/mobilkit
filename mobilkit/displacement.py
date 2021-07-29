# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

'''Tools and functions to analyze the displacement based on pings rather than
their localized tile IDs.
'''
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Polygon

from dask import dataframe as dd
from dask import array as da
import dask.bag as db

from mobilkit.spatial import haversine_pairwise, rad_of_gyr, total_distance_traveled
from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)

def process_user_displacement_pings(g):
    '''
    Given all the events of a user `uid` it computes the average and
    min distance from home, the closest point to home plus the original
    home location of the user for each `date` observed.
    
    Parameters
    ----------
    g : grouped df or pd.DataFrame
        A group of all the events of a user `uid` recorded in all the `dates`
        or dataframe as returned by
        :attr:`mobilkit.temporal.filter_daynight_time`
        and joined with a `uid,homelat,homelon` df with at least the
        `uid,date,homelat,homelon,lat,lng`
    
    Returns
    -------
    df : pd.DataFrame
        A df containing the original `uid,homelat,homelon`
        columns plus the minimum and average distance from home
        `mindist,avgdist` recorded on that `date` plus the `lat,lng`
        coordinates of the closest point to home for each observed day.
    '''
    g["date_group"] = g["date"].copy()
    return g.groupby("date_group").apply(process_user_day_displacement_pings)

def process_user_day_displacement_pings(g):
    '''
    Given all the events of a (user,date) it computes the average and
    min distance from home, the closest point to home plus the original
    home location of the user.
    
    Parameters
    ----------
    g : grouped df or pd.DataFrame
        A group of all the events of a user `uid` recorded in a day `date`
        or dataframe as returned by
        :attr:`mobilkit.temporal.filter_daynight_time` and joined with a
        `uid,homelat,homelon` df with at least the `uid,date,homelat,homelon,lat,lng`
    
    Returns
    -------
    df : pd.DataFrame
        A one-row df containing the original `uid,date,homelat,homelon`
        columns plus the minimum and average distance from home
        `mindist,avgdist` recorded on that `date` plus the `lat,lng`
        coordinates of the closest point to home.
    '''
    tmp_user, tmp_day = g[uidColName].iloc[0], g["date"].iloc[0]
    home_lat_lon = g[["homelat","homelon"]].iloc[:1].values
    tmp_coordinates = g[[latColName,lonColName]].values
    # Use the spatial fuinction to get distances in km
    distances = haversine_pairwise(tmp_coordinates, home_lat_lon)
    
    avg_dist = np.mean(distances[:,0])
    min_dist_idx = np.argmin(distances[:,0])
    min_dist = distances[min_dist_idx,0]
    min_dist_lat_lon = tmp_coordinates[min_dist_idx,:]
    
    # Radius of gyration and ttd
    rog = rad_of_gyr(tmp_coordinates)
    ttd = total_distance_traveled(tmp_coordinates)
    
    return pd.DataFrame([[
                        tmp_user, tmp_day,
                        home_lat_lon[0,0], home_lat_lon[0,1],
                        min_dist, avg_dist,
                        min_dist_lat_lon[0], min_dist_lat_lon[1],
                        rog, ttd
                         ]],
                       columns=[uidColName,"date",
                                "homelat","homelon",
                                "mindist","avgdist",
                                latColName,lonColName,
                                "rg", "ttd",
                               ])


def calc_displacement(df_usr_day, df_usr_home):
    '''
    Given all the events of a (user,date) it computes the average and
    min distance from home, the closest point to home plus the original
    home location of the user on that day.
    
    Parameters
    ----------
    df_usr_day : dask.DataFrame
        A dataframe containing all the events as returned by
        :attr:`mobilkit.temporal.filter_daynight_time`.
        

    df_usr_home : pd.DataFrame or dask.DataFrame
        A `uid,homelat,homelon` df containing the lat and lon of the user's
        home to be joined with `df_usr_day`.
        Note that only events belonging to `uid` contained in this df will
        be considered.
    
    Returns
    -------
    df : dask.DataFrame
        A df containing, for each `uid,date` couple observed:
        
        - `mindist,avgdist` the minimum and average distance from home recorded
          on that `date`
        - `lat,lng` coordinates of the closest point to home recorded on `date`
        - `homelat,homelon` the user's home coordinates
        - `rg,ttd` the radius of gyration and the total traveled distance for a
           user on `date`.
    '''
    
    # We perform a large to small join
    if type(df_usr_home) == pd.DataFrame:
        home_indexed = dd.from_pandas(df_usr_home, npartitions=1)
    else:
        home_indexed = df_usr_home
    home_indexed = home_indexed.repartition(npartitions=1)
    home_indexed = home_indexed[[uidColName,"homelat","homelon"]]
    all_joined = df_usr_day.merge(home_indexed, on=uidColName, how="inner")

    meta_return = dict(**all_joined.dtypes)
    meta_return = {k: meta_return[k]
                       for k in [uidColName,"date", "homelat","homelon"]}
    meta_return["mindist"] = np.float64
    meta_return["avgdist"] = np.float64
    meta_return[latColName] = np.float64
    meta_return[lonColName] = np.float64
    meta_return['rg'] = np.float64
    meta_return['ttd'] = np.float64

    return all_joined.groupby(uidColName)\
                .apply(process_user_displacement_pings,
                       meta=meta_return).reset_index(drop=True)