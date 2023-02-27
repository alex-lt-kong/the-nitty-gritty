# Timezone-aware datetime (with DST properly handled)

* Naive datetime from one timezone (`src_tz`, e.g., 'Asia/Urumqi',
'Europe/Luxembourg') to another timezone (`dst_tz`, e.g., 'Europe/Warsaw',
'Asia/Manila'):
  ```Python
  unix_ts = pytz.timezone(src_tz).localize(naive_dt).timestamp()
  dt.datetime.fromtimestamp(unix_ts, pytz.timezone(dst_tz)).replace(tzinfo=None)
  ```
  * Note that for Python the datetime object is always naive--`_src_tz` and
  `dst_tz` is something we know in advance but something doesn't know by
  Python's datetime object.

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
  * Any dates old than 1970-01-01 should be considered to be too far away.