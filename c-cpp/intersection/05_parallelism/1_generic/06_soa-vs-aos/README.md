# Structure of Arrays vs Array of Structures

* The general principle is that structure of arrays performs much better than array of structures.

* Under vanilla settings, both `icc` and `gcc` can't automatically vectorize the code. After adding hints to
`icc` and `gcc`, they can successfully vectorize both soa and aos functions.

* Results:
  * `icc`
  ```
  SoA w/  vectorization: avg:  75.60ms, std: 7800.71
  SoA w/o vectorization: avg: 146.33ms, std: 9659.94
  AoS w/  vectorization: avg: 274.06ms, std: 7605.32
  AoS w/o vectorization: avg: 278.09ms, std: 8914.05
  ```
  * `gcc`
  ```
  SoA w/  vectorization: avg:  74.75ms, std: 7488.92
  SoA w/o vectorization: avg: 162.48ms, std: 16860.44
  AoS w/  vectorization: avg: 284.74ms, std: 9477.58
  AoS w/o vectorization: avg: 282.07ms, std: 6494.34
  ```