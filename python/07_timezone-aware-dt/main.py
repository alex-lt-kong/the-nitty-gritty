# Timezone aware datetime conversion proves to be tricky even in Python
# Here is a relatively simple example of how to do it
import datetime as dt
import pytz


def test(naive_datetime: dt.datetime) -> None:
    # Say we are given a datetime object that is not timezone-aware.
    # We know that it is in a specific timezone, but we want it to be in another timezone.

    print(
        f'The navie_datetime is {naive_datetime}. '
        'Python is not aware of the timezone, but we know it is in UTC+8 (Asia/Hong_Kong).'
    )
    # We can get the Unix epoch time for the given naive/timezone-agnostic datetime object
    unix_ts = pytz.timezone('Asia/Hong_Kong').localize(naive_datetime).timestamp()

    print(f'It is equivalent to unix timestamp: {unix_ts}')

    ny_localtime = dt.datetime.fromtimestamp(unix_ts, pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Translating from unix timestamp to EST (US/Eastern) is relatively straightforward: {ny_localtime}')


test(dt.datetime(2020, 7, 1, 12, 34, 56))
print()
test(dt.datetime(2022, 12, 1, 6, 54, 32))
print(
    'Note that the difference between US/Eastern and Asia/Hong_Kong is dynamic--'
    'as US/Eastern follows DST while Asia/Hong_Kong does not.'
)
test(dt.datetime(1903, 1, 4, 15, 9, 26))
print('Also note that this function seems to be unable to handle dates before 1970.')


def incorrect_method(dt_str: str) -> None:
    # If we are given a string of a specific timezone, we may be tempted to directly convert it into unix_datetime
    # instead of using unix timestamp as a bridge.
    # This would not be as straightforward.
    naive_datetime = dt.datetime.strptime(dt_str, '%b %d %Y %I:%M:%S:%f%p')
    unix_dt = pytz.timezone('US/Eastern').localize(naive_datetime)
    # pytz.timezone('US/Eastern').localize(naive_datetime) produces a timezone-aware datetime
    # if we just print(), if will still show the original datetime, with timezone info
    print(unix_dt)
    # >>> 2022-12-01 20:53:05-05:00
    print(unix_dt.strftime('%Y-%m-%d %H:%M:%S'))
    # >>> 2022-12-01 20:53:05
    # This could be the more proper behavior after thinking it through, but this could be what we don't want:
    # we may just want to convert a datetime string in one timezone to a datetime string to another.


incorrect_method('Dec  1 2022  8:53:05:000PM')
