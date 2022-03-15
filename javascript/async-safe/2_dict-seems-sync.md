* `longOperation(i, true);`
```
0 { Apr: 0, Feb: 1, Aug: 1, Jul: 1, Dec: 1, Jan: 1 }
1 { Apr: 0, Feb: 0, Aug: 1, Jul: 1, Dec: 1, Jan: 1 }
2 { Apr: 0, Feb: -1, Aug: 1, Jul: 1, Dec: 1, Jan: 1 }
3 { Apr: 0, Feb: -1, Aug: 0, Jul: 1, Dec: 1, Jan: 1 }
4 { Apr: 0, Feb: -1, Aug: 0, Jul: 0, Dec: 1, Jan: 1 }
5 { Apr: 0, Feb: -1, Aug: 0, Jul: 0, Dec: 0, Jan: 1 }
6 { Apr: 0, Feb: -1, Aug: -1, Jul: 0, Dec: 0, Jan: 1 }
7 { Apr: 0, Feb: -1, Aug: -1, Jul: 0, Dec: 0, Jan: 0 }
8 { Apr: 0, Feb: -1, Aug: -1, Jul: -1, Dec: 0, Jan: 0 }
9 { Apr: 0, Feb: -1, Aug: -1, Jul: -1, Dec: -1, Jan: 0 }
```

* `longOperation(i, false);`
```
0 { Aug: 3, May: 2, Feb: 2, Apr: 1, Jun: 1 }
1 { Aug: 3, May: 1, Feb: 2, Apr: 1, Jun: 1 }
2 { Aug: 3, May: 1, Feb: 1, Apr: 1, Jun: 1 }
3 { Aug: 2, May: 1, Feb: 1, Apr: 1, Jun: 1 }
4 { Aug: 2, May: 0, Feb: 1, Apr: 1, Jun: 1 }
5 { Aug: 1, May: 0, Feb: 1, Apr: 1, Jun: 1 }
6 { Aug: 1, May: 0, Feb: 1, Apr: 0, Jun: 1 }
7 { Aug: 1, May: 0, Feb: 1, Apr: 0, Jun: 0 }
8 { Aug: 0, May: 0, Feb: 1, Apr: 0, Jun: 0 }
9 { Aug: 0, May: 0, Feb: 0, Apr: 0, Jun: 0 }
```