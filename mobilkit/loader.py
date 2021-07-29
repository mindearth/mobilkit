# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

'''`loader.py` contains a set of tools to load and prepare the database from raw files.
'''
import pytz

import pandas as pd
import numpy as np
import os
import sys

from datetime import datetime, timedelta
from glob import glob

from dask import dataframe as dd
from dask import bag as db
import pytz

from mobilkit.dask_schemas import eventLineRAW

from mobilkit.dask_schemas import (
    accColName,
    lonColName,
    latColName,
    uidColName,
    utcColName,
    dttColName,
    zidColName,
)

def load_raw_files(pattern, version="hflb",
                   timezone=None, start_date=None, stop_date=None,
                   minAcc=300, sep="\t", file_schema=None, **kwargs):
    '''Function that loads the files and returns the dataframe. 

    Parameters
    ----------
    pattern : str
        The pattern of the raw files with bash syntax. For example: ``'sample_data/20*/part-*.csv.gz'``.
    version : str, optional
        One of `hflb`, `wb` or `csv`, the format in which data are stored.
    timezone : str, optional
        The timezone in pytz syntax (e.g., "Europe/Rome" or "America/Mexico_City") to be
        used to localize the Unix Time Stamp time-stamp in the raw-data. If no timezone is specified (default) it defaults to UTC.
    start_date : str, optional
        The starting date when to consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given.
    stop_date : str, optional
        The end day up to which consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given. This day will be INCLUDED.
    minAcc : int, optional
        The minimum accuracy for a point to be kept. If accuracy is larger than this the point will be discarded. **NOTE that the accuracy column must be called** ``acc``.
    sep : str, optional
        The delimiter of the fields in the files.
    file_schema : list of tuples
        The schema of the file. By default will be :attr:`dask_schemas.eventLineRAW`.
        If `version=wb` `file_schema` is a dictionary telling how to translate the
        original colums in the mobilkit nomenclature.
        **NOTE that the accuracy column must be called** ``acc``.
    **kwargs
        Will be passed to :attr:`mobilkit.loader.load_raw_files_hflb`
        if `version='hflb'`  otherwise to :attr:`mobilkit.loader.load_raw_files_wb`
        if `version='wb'`.

    Returns
    -------
    df : dask.dataframe
        A representation of the dataframe.
    '''
    assert version in ["hflb","wb", "csv"]
    if version == "hflb":
        return load_raw_files_hflb(pattern=pattern, timezone=timezone,
                         start_date=start_date, stop_date=stop_date,
                         minAcc=minAcc, sep=sep, file_schema=file_schema, **kwargs)
    elif version == "wb":
        return load_raw_files_wb(pattern=pattern, timezone=timezone,
                         start_date=start_date, stop_date=stop_date,
                         minAcc=minAcc, sep=sep, file_schema=file_schema, **kwargs)
    elif version == "csv":
        return load_raw_files_custom(pattern=pattern, timezone=timezone,
                         start_date=start_date, stop_date=stop_date,
                         minAcc=minAcc, sep=sep, file_schema=file_schema, **kwargs)
    else:
        return None

def load_raw_files_custom(pattern, timezone=None,
                          start_date=None, stop_date=None,
                          minAcc=300, sep="\t", file_schema=None, partition_size=None, **kwargs):
    '''Function that loads the files and returns the dask dataframe. Note that this
    function is **lazy** meaning that it only construct the dataframe and does not
    build it (it will be built the first time a query is performed on it).

    Parameters
    ----------
    pattern : str
        The pattern of the raw files with bash syntax. For example: ``'sample_data/20*/part-*.csv.gz'``.
    timezone : str, optional
        The timezone in pytz syntax (e.g., "Europe/Rome" or "America/Mexico_City") to be
        used to localize the Unix Time Stamp time-stamp in the raw-data. If no timezone is specified (default) it defaults to UTC.
    start_date : str, optional
        The starting date when to consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given.
    stop_date : str, optional
        The end day up to which consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given. This day will be INCLUDED.
    minAcc : int, optional
        The minimum accuracy for a point to be kept. If accuracy is larger than this the point will be discarded. **NOTE that the accuracy column must be called** ``acc``.
    sep : str, optional
        The delimiter of the fields in the files.
    file_schema : list of tuples
        The schema of the file. By default will be :attr:`dask_schemas.eventLineRAW`.
        **NOTE that the accuracy column must be called** ``acc``.

    Returns
    -------
    df : dask.dataframe
        A representation of the dataframe.
    '''

    if file_schema is None:
        file_schema = eventLineRAW
    def tmp_filter(r):
        if len(r) != len(file_schema): return False
        for e in r:
            if len(e) == 0:
                return False
        return True
    sp_df = db.read_text(pattern)\
            .str.strip().str.split(sep)\
            .filter(tmp_filter)\
            .to_dataframe(file_schema)
    if partition_size is not None:
        sp_df = sp_df.repartition(partition_size=partition_size)
    if timezone is None:
        timezone = "UTC"
    selected_tz = pytz.timezone(timezone)
    sp_df = filterStartStopDates(sp_df, start_date, stop_date, selected_tz)
    sp_df = compute_datetime_col(sp_df, timezone)
    if minAcc is not None and minAcc > 0:
        print("Filtering on accuracy <= %d" % minAcc)
        sp_df = sp_df[sp_df[accColName] <= minAcc]
    return sp_df

def load_raw_files_hflb(pattern, timezone=None,
                          start_date=None, stop_date=None,
                          minAcc=300, sep="\t", file_schema=None, partition_size=None):
    '''Function that loads the files and returns the dask dataframe. Note that this
    function is **lazy** meaning that it only construct the dataframe and does not
    build it (it will be built the first time a query is performed on it).

    Parameters
    ----------
    pattern : str
        The pattern of the raw files with bash syntax. For example: ``'sample_data/20*/part-*.csv.gz'``.
    timezone : str, optional
        The timezone in pytz syntax (e.g., "Europe/Rome" or "America/Mexico_City") to be
        used to localize the Unix Time Stamp time-stamp in the raw-data. If no timezone is specified (default) it defaults to UTC.
    start_date : str, optional
        The starting date when to consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given.
    stop_date : str, optional
        The end day up to which consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given. This day will be INCLUDED.
    minAcc : int, optional
        The minimum accuracy for a point to be kept. If accuracy is larger than this the point will be discarded. **NOTE that the accuracy column must be called** ``acc``.
    sep : str, optional
        The delimiter of the fields in the files.
    file_schema : list of tuples
        The schema of the file. By default will be :attr:`dask_schemas.eventLineRAW`.
        **NOTE that the accuracy column must be called** ``acc``.

    Returns
    -------
    df : dask.dataframe
        A representation of the dataframe.
    '''

    if file_schema is None:
        file_schema = eventLineRAW
    
    def tmp_filter(r):
        if len(r) != len(file_schema): return False
        for e in r:
            if len(e) == 0:
                return False
        return True


    sp_df = db.read_text(pattern)\
            .str.strip().str.split(sep)\
            .filter(tmp_filter)\
            .to_dataframe(file_schema)

    if partition_size is not None:
        sp_df = sp_df.repartition(partition_size=partition_size)
    
    if timezone is None:
        timezone = "UTC"
    
    
    selected_tz = pytz.timezone(timezone)

    sp_df = filterStartStopDates(sp_df, start_date, stop_date, selected_tz)

    sp_df = compute_datetime_col(sp_df, timezone)
    
    if minAcc is not None and minAcc > 0:
        print("Filtering on accuracy <= %d" % minAcc)
        sp_df = sp_df[sp_df[accColName] <= minAcc]

    return sp_df


def load_raw_files_wb(pattern, timezone=None, header=False,
                          start_date=None, stop_date=None,
                          minAcc=300, sep="\t", file_schema=None, **kwargs):
    '''Function that loads the files and returns the dask dataframe. Note that this
    function is **lazy** meaning that it only construct the dataframe and does not
    build it (it will be built the first time a query is performed on it).

    Parameters
    ----------
    pattern : str
        The pattern of the raw files with bash syntax. For example: ``'sample_data/20*/part-*.csv.gz'``.
    timezone : str, optional
        The timezone in pytz syntax (e.g., "Europe/Rome" or "America/Mexico_City") to be
        used to localize the Unix Time Stamp time-stamp in the raw-data. If no timezone is specified (default) it defaults to UTC.
    start_date : str, optional
        The starting date when to consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given.
    stop_date : str, optional
        The end day up to which consider data in the "yyyy-mm-dd" format. This will be localized in `timezone` if given. This day will be INCLUDED.
    minAcc : int, optional
        The minimum accuracy for a point to be kept. If accuracy is larger than this the point will be discarded. **NOTE that the accuracy column must be called** ``acc``.
    sep : str, optional
        The delimiter of the fields in the files.
    file_schema : list of tuples
        The dict to rename the original columns to the mobilkit ones.
        **NOTE that the accuracy column must be called** ``acc``.

    Returns
    -------
    df : dask.dataframe
        A representation of the dataframe.
    '''
    files = [file for file in glob(pattern)]
    if header:
        sp_df = dd.read_csv(files, sep=sep)
    else:
        sp_df = dd.read_csv(files, names=file_schema.keys(), sep=sep)
    sp_df = sp_df.rename(columns=file_schema)
    selected_tz = pytz.timezone(timezone)
    sp_df = filterStartStopDates(sp_df, start_date, stop_date, selected_tz)
    sp_df = compute_datetime_col(sp_df, timezone)
    if minAcc is not None and minAcc > 0:
        print("Filtering on accuracy <= %d" % minAcc)
        sp_df = sp_df[sp_df[accColName] <= minAcc]
    return sp_df

def loaddata_takeapeek(dirpath, sep, ext):
    files = [file for file in glob(dirpath+"*"+ext)]
    for i,f in enumerate(files):
        print("filename of file #"+str(i)+": "+f)
        df = dd.read_csv(f, sep=sep)
        print("head of file #"+str(i)+":")
        print(df.head())

def compute_datetime_col(df, selected_tz):
    if dttColName in df.columns:
        print("Warning, %s column already present, not computing it...")
    else:
        df[dttColName] = dd.to_datetime(df[utcColName]*1e9, utc=False, unit="ns")\
                        .dt.tz_localize("UTC")\
                        .dt.tz_convert(selected_tz)\
                        .dt.tz_localize(None)

    return df

def localizeDatetimeNaive(date, tz, date_format="%Y-%m-%d"):
    dt = datetime.strptime(date, date_format)\
                        .astimezone(tz)\
                        .replace(tzinfo=None)
    return dt

def filterStartStopDates(df, start_date, stop_date, tz):
    if start_date is not None:
        start_UTC = localizeDatetimeNaive(start_date, tz).timestamp()
        print("Filtering on start UTC >= %d (%s)" % (start_UTC, start_date))
    else:
        start_UTC = -1
    if stop_date is not None:
        stop_UTC = localizeDatetimeNaive(stop_date, tz).timestamp()
        print("Filtering on stop UTC <= %d (%s)" % (stop_UTC, stop_date))
    else:
        stop_UTC = 99999999999999
    df = df[df[utcColName].between(start_UTC, stop_UTC)]
    return df

def load_from_skmob(df, uid="user", npartitions=10):
    '''
    Loads a dataframe imported with skmobility and returns a dask dataframe.

    Parameters
    ----------
    df : scikit-mobility.dataframe
        A dataframe as imported from scikit-mobility. May already contains the ``tile_ID`` and ``uid`` columns. If no ``uid`` column is found it will be initialized to the ``uid`` value.
    uid : str, optional
        The ``uid`` to be used, otherwise uses the present ones if the ``uid`` column is there.
    npartitions : int, optional
        The number of partition for the dataframe to be split into.

    Returns
    -------
    df_sp : dask.dataframe
        A `dask.dataframe` containing the input columns plus the accuracy ``acc`` (with dummy 1 value) and possibly the ``uid`` one if it was missing.


    '''
    import skmob
    # Create the acc column
    if accColName not in df.columns:
        df[accColName] = 1

    if uidColName not in df.columns:
        df[uidColName] = uid
    df_sp = dd.from_pandas(df, npartitions=npartitions)
    return df_sp

def dask_to_skmob(df, **kwargs):
    '''
    Ports a dataframe from dask to skmob. Given the structure of skmob it is done only to a `skmob.TrajDataFrame`.

    Parameters
    ----------
    df : dask.dataframe
        A dask dataframe with at least the ``uid``, ``lat`` and ``lng`` columns.
    **kwargs
        Will be passed to ``skmob.TrajDataFrame``.

    Returns
    -------
    df_sp : skmob.dataframe
        A skmob.TrajDataFrame containing the input columns.
    '''
    import skmob
    traj = skmob.TrajDataFrame(df.compute(), **kwargs)
    return traj

def persistDF(df, path, overwrite=True, header=True, index=False, out_format="csv"):
    '''
    Save a dask dataframo file.
    
    Parameters
    ----------
    df : dask.DataFrame
        The dataframe to save
    path : str
        The path where to save the dataframe.
    overwrite : bool
        Whether or not to force overwrite.
    header : bool
        Whether or not to put the header in the output file.
    index : bool
        Whether or not to put the index column in the output file.
    out_format : bool
        One of ``csv, parquet`` the format to use. If the df has arrays in it use ``parquet``.
    '''
    
    if not overwrite and os.path.exists(path):
        raise RuntimeError("Path %s already exist, force overwrite to continue." % path)

    if out_format == "csv":
        df.to_csv(path, header=header, index=index)
    elif out_format == "parquet":
        df.to_parquet(path, write_index=index, overwrite=overwrite)
    else:
        raise RuntimeError("Unknown format %s in persistDF" % out_format)

def reloadDF(path, header=True, in_format="csv"):
    '''
    Load a dask dataframe from file.
    
    Parameters
    ----------
    path : str
        The path where to read the dataframe from.
    header : bool
        Whether or not to read the header in the output file.
    in_format : bool
        One of ``csv, parquet`` the format used to persist the df.
    
    Returns
    -------
    df : dask.DataFrame
        The loaded dataframe.
    '''
    if in_format == "csv":
            df = dd.read_csv(path)
    elif in_format == "parquet":
            df = dd.read_parquet(path)
    else:
        raise RuntimeError("Unknown format %s in reloadDF" % out_format)
        
    return df

def fromunix2fulldate(x, timezone="America/Mexico_City"):
    '''
    Inherited from D4R.
    '''
    ct = pytz.timezone(timezone)
    ### errors 
    if (int(x)<86400) or (int(x)>1609518354):
        x = 86400
    dat = datetime.fromtimestamp(x, ct)
    tim = dat.strftime('%Y-%m-%d %H:%M:%S') ### date --> string
    tim2 = dat.strptime(tim, '%Y-%m-%d  %H:%M:%S') ### string --> date
    return tim2

def fromunix2date(x, timezone="America/Mexico_City"):
    '''
    Inherited from D4R.
    '''
    ct = pytz.timezone(timezone)
    ### errors 
    if (int(x)<86400) or (int(x)>1609518354):
        x = 86400
    dat = datetime.fromtimestamp(x, ct)
    tim = dat.strftime('%Y-%m-%d') ### date --> string
    tim2 = dat.strptime(tim, '%Y-%m-%d') ### string --> date
    return tim2

def fromunix2time(x, timezone="America/Mexico_City"):
    '''
    Inherited from D4R.
    '''
    ct = pytz.timezone(timezone)
    if (int(x)<86400) or (int(x)>1609518354):
        x = 86400
    dat = datetime.fromtimestamp(x, ct)
    tim = dat.strftime('%H:%M:%S')
    tim2 = dat.strptime(tim, '%H:%M:%S')
    return tim2

def crop_date(dff, startdt, enddt, timezone="America/Mexico_City"):
    ### input string start date 
    ct = pytz.timezone(timezone)
    st = int(datetime.strptime(startdt, "%Y/%m/%d").replace(tzinfo=ct).timestamp())
    et = int(datetime.strptime(enddt, "%Y/%m/%d").replace(tzinfo=ct).timestamp())
    df_res = dff[(dff[utcColName] <= et) & (dff[utcColName] >= st)]
    return df_res

def crop_time(dff, nighttime_start, nighttime_end, timezone): ### input string start date 
    ct = pytz.timezone(timezone)
    daytime1 =  datetime.strptime(nighttime_start, '%H:%M:%S')
    daytime2 =  datetime.strptime(nighttime_end, '%H:%M:%S')
    meta = dff[utcColName].head(1).apply(fromunix2time)
    dff["time_t"] = dff[utcColName].apply(lambda x: fromunix2time(x))
    dff_nighttime = dff[(dff["time_t"]<daytime1) | (dff["time_t"]>daytime2)]
    return dff_nighttime

def crop_spatial(dff, bbox):
    '''
    Filters `dff` with a `box=[minlon,minlat,maxlon,maxlat]`.
    '''
    minlon = bbox[0]
    minlat = bbox[1]
    maxlon = bbox[2]
    maxlat = bbox[3]
    df_res = dff[(dff[lonColName] <= maxlon) & (dff[lonColName] >= minlon) \
                 & (dff[latColName] <= maxlat) & (dff[latColName] >= minlat)]
    return df_res


def loadGeolifeData(path, acc_default=1, timezone="Asia/Shanghai"):
    '''
    Loads the `Geolife v1.3` trajectories with files ordered in the
    
        `Geolife\ Trajectories\ 1.3/Data/000/Trajectory/20090401202331.plt`
        
    structure with 6 useless rows at the beginning and the
    
        `lat,lng,0,altitude,days,date,time`
        
    format.
    
    Parameters
    ----------
    path : str
        The path to the root of the geolife data, usually called
        `data/Geolife\ Trajectories\ 1.3`.
    acc_default : float
        The default accuracy to give to each point to replicate the `mobilkit`
        format.
    timezone : str
        The code of the timezone the data has been recorded in.
    
    Returns
    -------
    df : pd.DataFrame
        The dataframe containing the
            `uid,UTC,datetime,acc,lat,lng` columns.
    '''
    
    folders_pattern = os.path.join(path, "Data/*")
    user_dirs = [d for d in sorted(glob(folders_pattern)) if os.path.isdir(d)]
    df = pd.DataFrame()
    n_users = len(user_dirs)
    sys.stdout.write("Found %d users...\n" % n_users)
    sys.stdout.flush()
    from multiprocessing import Pool, cpu_count
    world = cpu_count()
    with Pool(world) as p:
        dfs = p.map(_loaderGeoLifeBatch, [(user_dirs, acc_default, timezone, i, world) for i in range(world)])
    df = pd.concat(dfs, ignore_index=True, sort=True)
    sys.stdout.write("\nDone!")
    sys.stdout.flush()
    
    return df

def _loaderGeoLifeBatch(args):
    user_dirs, acc_default, timezone, rank, world = args
    df = pd.DataFrame()
    for user_dir in user_dirs[rank::world]:
        tmp_user = int(os.path.basename(user_dir))
        traj_pattern = os.path.join(user_dir, "Trajectory/*.plt")
        for traj in sorted(glob(traj_pattern)):
            if not os.path.isfile(traj):
                continue
            tmp_rows = pd.read_csv(traj,sep=",", skiprows=6, header=None)
            tmp_rows.columns = [latColName,lonColName,"OS","junk_1","junk_2","day","time"]
            tmp_rows["datetime_str"] = tmp_rows.apply(lambda r: r["day"] + " " + r["time"], axis=1)
            tmp_rows[uidColName] = [tmp_user]*tmp_rows.shape[0]
            tmp_rows[accColName] = [acc_default]*tmp_rows.shape[0]
            tmp_rows[dttColName] = pd.to_datetime(tmp_rows["datetime_str"], utc=True,
                                                  format="%Y-%m-%d %H:%M:%S")\
                                        .dt.tz_convert(timezone)

            tmp_rows[utcColName] = (tmp_rows[dttColName].dt.tz_convert("GMT")
                                    - pd.Timestamp("1970-01-01", tz="GMT"))\
                                            // pd.Timedelta('1s')

            tmp_rows = tmp_rows[[uidColName,utcColName,dttColName,accColName,latColName,lonColName]]
            df = pd.concat((df, tmp_rows), ignore_index=True, sort=True)
    sys.stdout.write("Done process %03d / %03d\n" % (rank+1, world))
    sys.stdout.flush()
    return df


def syntheticGeoLifeDay(df_geolife, selected_day):
    # Many users, one day aggregation
    # Each user/{weekday/weekend} is a different user
    df_users = df_geolife.copy()

    df_users["day"] = df_users[dttColName].dt.floor("1d")

    seen_couples = df_users.groupby([uidColName,"day"])[[]].agg("count").reset_index()
    target_col = uidColName + "_new"
    seen_couples[target_col] = np.arange(0, seen_couples.shape[0])
    df_users = pd.merge(df_users, seen_couples, on=[uidColName,"day"], how="inner")
    df_users[uidColName] = df_users[target_col].values
    del df_users[target_col], df_users["day"]

    df_users[dttColName] = selected_day + pd.to_timedelta(df_users[dttColName].dt.hour*3600
                                                            + df_users[dttColName].dt.minute*60
                                                            + df_users[dttColName].dt.second, unit="s")

    df_users[utcColName] = (df_users[dttColName].dt.tz_convert("GMT")
                                        - pd.Timestamp("1970-01-01", tz="GMT"))\
                                                // pd.Timedelta('1s')
    
    return df_users

def syntheticGeoLifeWeek(df_geolife, selected_week):
    while selected_week.weekday() > 0:
        selected_week -= timedelta(days=1)
        if selected_week.weekday() == 0:
            print("Anticipated the date to Monday: ", selected_week)
            
    # Less users, one week aggregation
    # Each user/week is a different user, all projected to a given week
    df_week = df_geolife.copy()

    df_week["week"] = df_week[dttColName].dt.floor("1d") - pd.to_timedelta(df_week[dttColName].dt.weekday, unit="d")

    seen_couples = df_week.groupby([uidColName,"week"])[[]].agg("count").reset_index()
    target_col = uidColName + "_new"
    seen_couples[target_col] = np.arange(0, seen_couples.shape[0])
    df_week = pd.merge(df_week, seen_couples, on=[uidColName,"week"], how="inner")
    df_week[uidColName] = df_week[target_col].values
    del df_week[target_col], df_week["week"]

    df_week[dttColName] = selected_week + pd.to_timedelta(df_week[dttColName].dt.weekday*(24*3600)
                                                            + df_week[dttColName].dt.hour*3600
                                                            + df_week[dttColName].dt.minute*60
                                                            + df_week[dttColName].dt.second, unit="s")

    df_week[utcColName] = (df_week[dttColName].dt.tz_convert("GMT")
                                        - pd.Timestamp("1970-01-01", tz="GMT"))\
                                                // pd.Timedelta('1s')
    
    return df_week
