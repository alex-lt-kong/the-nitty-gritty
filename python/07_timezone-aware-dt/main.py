import datetime as dt
import pytz
import time


def convert_naive_dt(naive_dt: dt.datetime, src_tz: str, dst_tz: str) -> dt.datetime:
    # Say we are given a datetime object that is not timezone-aware.
    # We know that it is in a specific timezone, but we want it to be
    # in another timezone.

    # We can get the Unix epoch time (in UTC, of course) for the
    # given naive(i.e., timezone-agnostic) datetime object
    unix_ts = pytz.timezone(src_tz).localize(naive_dt).timestamp()

    # Translating from unix timestamp to dst_tz is relatively straightforward
    dst_localtime = dt.datetime.fromtimestamp(
        unix_ts, pytz.timezone(dst_tz)
    ).replace(tzinfo=None)
    # replace(tzinfo) drops timezone awareness.
    return dst_localtime


def get_local_today(tz: str) -> dt.date:
    return dt.datetime.fromtimestamp(time.time(), pytz.timezone(tz)).date()


def test_conversions() -> None:
    def assert_with_message(
            actual_dt: dt.datetime, expect_dt: dt.datetime
    ) -> None:
        if actual_dt != expect_dt:
            raise ValueError(f'actual: {actual_dt}\nexpect: {expect_dt}')

    # DST is in effect in US/Eastern but not in Asia/Hong_Kong
    src_tz, src_dt = 'Asia/Hong_Kong', dt.datetime(2020, 7, 1, 12, 34, 56)
    dst_tz, dst_dt = 'US/Eastern', dt.datetime(2020, 7, 1, 0, 34, 56)
    assert_with_message(convert_naive_dt(src_dt, src_tz, dst_tz), dst_dt)

    # DST is not in effect in both
    src_tz, src_dt = 'Asia/Hong_Kong', dt.datetime(2022, 12, 1, 6, 54, 32)
    dst_tz, dst_dt = 'US/Eastern', dt.datetime(2022, 11, 30, 17, 54, 32)
    assert_with_message(convert_naive_dt(src_dt, src_tz, dst_tz), dst_dt)

    # DST is in effect in both
    src_tz, src_dt = 'US/Eastern', dt.datetime(2028, 5, 10, 3, 3, 3)
    dst_tz, dst_dt = 'Europe/London', dt.datetime(2028, 5, 10, 8, 3, 3)
    assert_with_message(convert_naive_dt(src_dt, src_tz, dst_tz), dst_dt)

    # DST is NOT in effect in both
    src_tz, src_dt = 'Asia/Tokyo', dt.datetime(2014, 4, 9, 12, 0, 0)
    dst_tz, dst_dt = 'Asia/Hong_Kong', dt.datetime(2014, 4, 9, 11, 0, 0)
    assert_with_message(convert_naive_dt(src_dt, src_tz, dst_tz), dst_dt)


def incorrect_method(dt_str: str) -> None:
    # If we are given a string of a specific timezone, we may be tempted to
    # directly convert it into unix_datetime instead of using unix timestamp
    # as a bridge. This would not be as straightforward.
    naive_datetime = dt.datetime.strptime(dt_str, '%b %d %Y %I:%M:%S:%f%p')
    unix_dt = pytz.timezone('US/Eastern').localize(naive_datetime)
    # pytz.timezone('US/Eastern').localize(naive_datetime) produces a
    # timezone-aware datetime if we just print(), if will still show the
    # original datetime, with timezone info
    print(unix_dt)
    # >>> 2022-12-01 20:53:05-05:00
    print(unix_dt.strftime('%Y-%m-%d %H:%M:%S'))
    # >>> 2022-12-01 20:53:05
    # This could be the more proper behavior after thinking it through,
    # but this could be what we don't want: we may just want to convert a
    # datetime string in one timezone to a datetime string to another.


def main() -> None:
    test_conversions()
    incorrect_method('Dec  1 2022  8:53:05:000PM')


if __name__ == '__main__':
    main()
