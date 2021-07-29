# Copyright (C) MindEarth <enrico.ubaldi@mindearth.org> @ Mindearth 2020-2021
# 
# This file is part of mobilkit.
#
# mobilkit is distributed under the MIT license.

''' The typical types of the dataframe being processed and the fixed column names.
'''

from datetime import datetime

from dask import dataframe as dd

unique = dd.Aggregation(
    name="unique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
        finalize=lambda s1: s1.apply(lambda final: set(final)),
)

nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),
)

accColName = "acc"
lonColName = "lng"
latColName = "lat"
uidColName = "uid"
utcColName = "UTC"
dttColName = "datetime"
zidColName = "tile_ID"


eventLineRAW = [
    (utcColName, int),
    (uidColName, str),
    ("OS", int),
    (latColName, float),
    (lonColName, float),
    (accColName, int),
    ("timezone", int),
]
'''The default type for raw files.
It is composed by:

    - `UTC` : unix time stamp (int)
    - `uid` : unique user identifier (string)
    - `OS`  : The os of the user (int)
    - `lat` : latitude (float)
    - `lng` : longitude (float)
    - `acc` : accuracy (float)
    - `timezone`  : time zone offset (float)
'''

eventLineDT = [
    (utcColName, int),
    (uidColName, str),
    ("OS", int),
    (latColName, float),
    (lonColName, float),
    (accColName, int),
    ("timezone", int),
    (dttColName, datetime),
]
'''The default type for raw files with datetime added.
It is composed by:

    - `UTC` : unix time stamp (int)
    - `UID` : unique user identifier (string)
    - `OS`  : The os of the user (int)
    - `lat` : latitude (float)
    - `lon` : longitude (float)
    - `acc` : accuracy (float)
    - `tz`  : time zone offset (float)
    - `date`  : datetime (datetime)
'''

eventLineZone = [
    (utcColName, int),
    (uidColName, str),
    ("OS", float),
    (latColName, float),
    (lonColName, float),
    (accColName, float),
    ("timezone", float),
    (zidColName, int),
]
'''The default type for raw files with zone index (int) added.
It is composed by:

    - `UTC` : unix time stamp (int)
    - `UID` : unique user identifier (string)
    - `OS`  : The os of the user (int)
    - `lat` : latitude (float)
    - `lon` : longitude (float)
    - `acc` : accuracy (float)
    - `tz`  : time zone offset (float)
    - `ZONE_IDX`  : the index of the containing area (int, -1 if outside of shapefile)
'''

eventLineDTzone = [
    (utcColName, int),
    (uidColName, str),
    ("OS", int),
    (latColName, float),
    (lonColName, float),
    (accColName, int),
    ("timezone", int),
    (dttColName, datetime),
    (zidColName, int),
]
'''The default type for raw files with datetime and zone index (int) added.
It is composed by:

    - `UTC` : unix time stamp (int)
    - `UID` : unique user identifier (string)
    - `OS`  : The os of the user (int)
    - `lat` : latitude (float)
    - `lon` : longitude (float)
    - `acc` : accuracy (float)
    - `tz`  : time zone offset (float)
    - `date`  : datetime (datetime)
    - `ZONE_IDX`  : the index of the containing area (int, -1 if outside of shapefile)
'''
