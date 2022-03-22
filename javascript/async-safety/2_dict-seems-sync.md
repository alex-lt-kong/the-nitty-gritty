* `longOperation(i, true);`
```
0 { Jun: 1, Nov: 0, Sep: 0, Jul: 0, Dec: 0, Jan: 0, Oct: 0 }
1 { Jun: 1, Nov: 1, Sep: 0, Jul: 0, Dec: 0, Jan: 0, Oct: 0 }
2 { Jun: 1, Nov: 1, Sep: 1, Jul: 0, Dec: 0, Jan: 0, Oct: 0 }
3 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 0, Jan: 0, Oct: 0 }
4 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
5 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
6 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 1, Oct: 0 }
7 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 1, Oct: 0 }
8 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 1, Oct: 0 }
9 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 1, Oct: 1 }
6 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 0, Oct: 1 }
9 { Jun: 1, Nov: 1, Sep: 1, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
1 { Jun: 1, Nov: 0, Sep: 1, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
2 { Jun: 1, Nov: 0, Sep: 0, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
0 { Jun: 0, Nov: 0, Sep: 0, Jul: 1, Dec: 1, Jan: 0, Oct: 0 }
5 { Jun: 0, Nov: 0, Sep: 0, Jul: 0, Dec: 1, Jan: 0, Oct: 0 }
8 { Jun: 0, Nov: -1, Sep: 0, Jul: 0, Dec: 1, Jan: 0, Oct: 0 }
7 { Jun: 0, Nov: -1, Sep: 0, Jul: 0, Dec: 1, Jan: -1, Oct: 0 }
4 { Jun: 0, Nov: -1, Sep: 0, Jul: 0, Dec: 0, Jan: -1, Oct: 0 }
3 { Jun: 0, Nov: -1, Sep: 0, Jul: -1, Dec: 0, Jan: -1, Oct: 0 }

```

* `longOperation(i, false);`
```
0 { Jun: 1 }
1 { Jun: 1, May: 1 }
2 { Jun: 1, May: 2 }
3 { Jun: 1, May: 3 }
4 { Jun: 1, May: 4 }
5 { Jun: 1, May: 4, Jul: 1 }
6 { Jun: 1, May: 4, Jul: 1, Apr: 1 }
7 { Jun: 1, May: 4, Jul: 1, Apr: 1, Mar: 1 }
8 { Jun: 1, May: 4, Jul: 1, Apr: 1, Mar: 2 }
9 { Jun: 1, May: 4, Jul: 1, Apr: 1, Mar: 2, Dec: 1 }
6 { Jun: 1, May: 4, Jul: 1, Apr: 0, Mar: 2, Dec: 1 }
5 { Jun: 1, May: 4, Jul: 0, Apr: 0, Mar: 2, Dec: 1 }
9 { Jun: 1, May: 4, Jul: 0, Apr: 0, Mar: 2, Dec: 0 }
0 { Jun: 0, May: 4, Jul: 0, Apr: 0, Mar: 2, Dec: 0 }
3 { Jun: 0, May: 3, Jul: 0, Apr: 0, Mar: 2, Dec: 0 }
8 { Jun: 0, May: 3, Jul: 0, Apr: 0, Mar: 1, Dec: 0 }
2 { Jun: 0, May: 2, Jul: 0, Apr: 0, Mar: 1, Dec: 0 }
4 { Jun: 0, May: 1, Jul: 0, Apr: 0, Mar: 1, Dec: 0 }
1 { Jun: 0, May: 0, Jul: 0, Apr: 0, Mar: 1, Dec: 0 }
7 { Jun: 0, May: 0, Jul: 0, Apr: 0, Mar: 0, Dec: 0 }

```