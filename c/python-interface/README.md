* Python's C interface seems to be pretty straightforward to use.
* The performance, per the below naive tests, seems to be "exactly native", not just "near native":

```
c   avg result: 176 msec
cpy avg result: 179 msec
np  avg result: 138 msec
py  avg result: 15,747 msec
```