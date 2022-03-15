* `var myDict = new NaiveDict(true);`
```
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ <10 empty items> ] [ <10 empty items> ]
[ 'Jan', <9 empty items> ] [ 0, <9 empty items> ]
[ 'Jan', 'Apr', <8 empty items> ] [ 0, 0, <8 empty items> ]
[ 'Jan', 'Apr', 'Apr', <7 empty items> ] [ 0, -1, 0, <7 empty items> ]
[ 'Jan', 'Apr', 'Apr', 'Apr', <6 empty items> ] [ 0, -2, -1, 0, <6 empty items> ]
[ 'Jan', 'Apr', 'Apr', 'Apr', 'Feb', <5 empty items> ] [ 0, -2, -1, 0, 0, <5 empty items> ]
[ 'Jan', 'Apr', 'Apr', 'Apr', 'Feb', 'Feb', <4 empty items> ] [ 0, -2, -1, 0, -1, 0, <4 empty items> ]
[ 'Jan', 'Apr', 'Apr', 'Apr', 'Feb', 'Feb', 'Mar', <3 empty items> ] [ 0, -2, -1, 0, -1, 0, 0, <3 empty items> ]
[
  'Jan',
  'Apr',
  'Apr',
  'Apr',
  'Feb',
  'Feb',
  'Mar',
  'Jan',
  <2 empty items>
] [ -1, -2, -1, 0, -1, 0, 0, 0, <2 empty items> ]
[
  'Jan',
  'Apr',
  'Apr',
  'Apr',
  'Feb',
  'Feb',
  'Mar',
  'Jan',
  'May',
  <1 empty item>
] [ -1, -2, -1, 0, -1, 0, 0, 0, 0, <1 empty item> ]
[
  'Jan', 'Apr', 'Apr',
  'Apr', 'Feb', 'Feb',
  'Mar', 'Jan', 'May',
  'May'
] [
  -1, -2, -1,  0, -1,
   0,  0,  0, -1,  0
]
```

* `var myDict = new NaiveDict(false);`
```
[ 'Jan', <9 empty items> ] [ 1, <9 empty items> ]
[ 'Jan', <9 empty items> ] [ 2, <9 empty items> ]
[ 'Jan', <9 empty items> ] [ 3, <9 empty items> ]
[ 'Jan', 'Mar', <8 empty items> ] [ 3, 1, <8 empty items> ]
[ 'Jan', 'Mar', 'Apr', <7 empty items> ] [ 3, 1, 1, <7 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 3, 1, 1, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 3, 1, 2, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 4, 1, 2, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 4, 1, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 5, 1, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 4, 1, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 3, 1, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 2, 1, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 2, 0, 2, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 2, 0, 1, 2, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 2, 0, 1, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 2, 0, 0, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 1, 0, 0, 1, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 1, 0, 0, 0, <6 empty items> ]
[ 'Jan', 'Mar', 'Apr', 'May', <6 empty items> ] [ 0, 0, 0, 0, <6 empty items> ]
```