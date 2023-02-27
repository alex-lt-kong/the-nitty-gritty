# Timezone-aware datetime (with DST properly handled)

* To convert a naive datetime `naive_dt` from `src_tz` timezone (e.g.,
'Asia/Urumqi', 'Europe/Luxembourg') to `dst_tz` timezone (e.g., 'Europe/Warsaw',
'Asia/Manila'):
  ```Python
  unix_ts = pytz.timezone(src_tz).localize(naive_dt).timestamp()
  dt.datetime.fromtimestamp(unix_ts, pytz.timezone(dst_tz)).replace(tzinfo=None)
  ```
  * Note that for Python the datetime object is always naive--`src_tz` and
  `dst_tz` are something we know in advance but something doesn't know by
  Python's datetime object.
  * Not letting datetime objects know timezone info is usually beneficial
  as it could cause confusion, especially when datetime objects are shared by
  many functions.

* Get "today" of timezone `tz` (e.g., 'America/Sao_Paulo', 'Asia/Tokyo').
  ```Python
  dt.datetime.fromtimestamp(time.time(), pytz.timezone(tz)).date()
  ```

* Strings like 'Asia/Urumqi', 'Europe/Warsaw', 'Asia/Tokyo'mentioned above are
called "TZ database name" as defined in TZ database (a.k.a., the zoneinfo
database or IANA time zone database, and occasionally as the Olson database) 
[here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

## Caveat

* Timezones are just like laws--they change, sometimes pretty frequently. As
a result, methods covered in this repo are most likely to be only correct for
dates that are not "too far away from now".
  * Any dates older than 1970-01-01 should be considered to be "too far away
  from now".