```
# ./main-vec-on.out 
add_one_func:
21ms,24ms,21ms,22ms,22ms,21ms,23ms,22ms,21ms,23ms,21ms,22ms,22ms,22ms,22ms,22ms,22ms,23ms,22ms,22ms,23ms,23ms,23ms,21ms,26ms,22ms,22ms,22ms,23ms,21ms,22ms,25ms,
23ms,23ms,25ms,24ms,24ms,22ms,23ms,22ms,22ms,23ms,23ms,24ms,24ms,23ms,22ms,25ms,23ms,24ms,24ms,23ms,22ms,22ms,23ms,24ms,24ms,23ms,23ms,23ms,26ms,23ms,23ms,22ms,
24ms,23ms,23ms,25ms,22ms,22ms,21ms,21ms,36ms,36ms,24ms,23ms,22ms,25ms,22ms,22ms,24ms,23ms,24ms,23ms,23ms,23ms,23ms,23ms,24ms,24ms,22ms,24ms,23ms,23ms,22ms,23ms,
23ms,24ms,23ms,23ms,23ms,24ms,22ms,22ms,22ms,22ms,22ms,25ms,22ms,22ms,22ms,22ms,22ms,22ms,22ms,22ms,24ms,23ms,22ms,23ms,22ms,22ms,21ms,22ms,22ms,22ms,22ms,22ms,
Average: 22

linear_func:
28ms,29ms,27ms,25ms,30ms,25ms,26ms,27ms,26ms,26ms,27ms,28ms,28ms,26ms,26ms,27ms,26ms,25ms,27ms,26ms,27ms,27ms,26ms,27ms,26ms,26ms,28ms,26ms,41ms,38ms,26ms,26ms,
25ms,26ms,25ms,26ms,28ms,27ms,25ms,26ms,27ms,25ms,26ms,25ms,25ms,26ms,28ms,26ms,26ms,27ms,28ms,27ms,25ms,26ms,30ms,26ms,27ms,26ms,26ms,26ms,27ms,28ms,28ms,25ms,
27ms,26ms,27ms,27ms,26ms,28ms,28ms,26ms,26ms,26ms,27ms,28ms,25ms,26ms,25ms,25ms,26ms,25ms,26ms,27ms,26ms,25ms,26ms,26ms,26ms,25ms,26ms,26ms,25ms,25ms,26ms,26ms,
26ms,26ms,27ms,28ms,25ms,27ms,27ms,26ms,26ms,37ms,43ms,27ms,27ms,27ms,28ms,27ms,26ms,25ms,25ms,27ms,25ms,26ms,26ms,26ms,26ms,29ms,26ms,27ms,27ms,27ms,28ms,27ms,
Average: 26

quadratic_func:
36ms,38ms,36ms,36ms,37ms,36ms,39ms,36ms,37ms,36ms,35ms,37ms,36ms,37ms,37ms,36ms,36ms,37ms,35ms,36ms,36ms,36ms,37ms,36ms,35ms,38ms,37ms,35ms,38ms,36ms,37ms,36ms,
35ms,37ms,36ms,36ms,36ms,36ms,35ms,65ms,36ms,37ms,35ms,36ms,35ms,36ms,36ms,36ms,35ms,37ms,36ms,35ms,35ms,37ms,40ms,37ms,38ms,36ms,36ms,36ms,36ms,36ms,38ms,35ms,
37ms,35ms,37ms,35ms,36ms,36ms,35ms,36ms,35ms,37ms,37ms,36ms,35ms,37ms,37ms,35ms,36ms,37ms,36ms,37ms,35ms,36ms,36ms,35ms,37ms,36ms,38ms,37ms,35ms,36ms,36ms,62ms,
35ms,36ms,35ms,37ms,36ms,35ms,37ms,35ms,36ms,36ms,37ms,38ms,36ms,35ms,36ms,35ms,37ms,36ms,35ms,36ms,35ms,36ms,37ms,35ms,36ms,37ms,35ms,35ms,36ms,36ms,36ms,36ms,
Average: 36
```

```
# ./main-vec-off.out 
add_one_func:
32ms,32ms,32ms,31ms,33ms,32ms,32ms,34ms,32ms,33ms,33ms,32ms,34ms,33ms,34ms,69ms,33ms,33ms,32ms,33ms,32ms,33ms,35ms,34ms,32ms,35ms,34ms,32ms,33ms,33ms,32ms,32ms,
32ms,32ms,32ms,32ms,32ms,32ms,32ms,32ms,33ms,32ms,31ms,32ms,31ms,32ms,32ms,31ms,32ms,32ms,34ms,32ms,33ms,33ms,34ms,33ms,34ms,33ms,34ms,33ms,34ms,33ms,33ms,33ms,
33ms,32ms,34ms,33ms,32ms,34ms,33ms,31ms,32ms,33ms,32ms,32ms,33ms,53ms,39ms,32ms,33ms,32ms,32ms,32ms,33ms,34ms,32ms,34ms,32ms,33ms,32ms,32ms,33ms,32ms,32ms,32ms,
33ms,32ms,32ms,32ms,33ms,33ms,33ms,33ms,32ms,34ms,33ms,33ms,33ms,32ms,32ms,33ms,33ms,33ms,32ms,31ms,32ms,32ms,33ms,32ms,32ms,31ms,32ms,33ms,33ms,31ms,32ms,33ms,
Average: 33

linear_func:
35ms,38ms,35ms,36ms,36ms,36ms,36ms,35ms,38ms,35ms,45ms,54ms,36ms,38ms,35ms,36ms,36ms,37ms,38ms,36ms,36ms,36ms,36ms,36ms,36ms,36ms,36ms,36ms,36ms,37ms,36ms,36ms,
36ms,38ms,36ms,35ms,36ms,36ms,37ms,35ms,36ms,36ms,36ms,36ms,37ms,38ms,37ms,35ms,36ms,36ms,35ms,38ms,35ms,35ms,36ms,36ms,35ms,36ms,36ms,36ms,36ms,35ms,36ms,35ms,
36ms,36ms,43ms,56ms,36ms,35ms,37ms,36ms,37ms,37ms,35ms,36ms,36ms,37ms,37ms,36ms,35ms,36ms,36ms,37ms,38ms,38ms,36ms,36ms,37ms,35ms,36ms,36ms,36ms,37ms,36ms,36ms,
36ms,37ms,36ms,36ms,36ms,37ms,36ms,37ms,38ms,36ms,36ms,36ms,35ms,37ms,36ms,36ms,36ms,35ms,37ms,37ms,36ms,37ms,37ms,36ms,36ms,37ms,57ms,44ms,35ms,37ms,36ms,36ms,
Average: 36

quadratic_func:
49ms,49ms,49ms,48ms,48ms,48ms,48ms,48ms,49ms,49ms,48ms,48ms,49ms,48ms,48ms,48ms,49ms,48ms,49ms,48ms,48ms,49ms,49ms,48ms,49ms,50ms,47ms,50ms,49ms,48ms,48ms,48ms,
50ms,49ms,48ms,49ms,48ms,71ms,54ms,49ms,48ms,48ms,48ms,50ms,48ms,48ms,48ms,48ms,49ms,48ms,48ms,48ms,49ms,48ms,48ms,48ms,47ms,50ms,48ms,49ms,48ms,47ms,48ms,48ms,
49ms,47ms,47ms,49ms,47ms,49ms,47ms,48ms,49ms,48ms,49ms,48ms,49ms,50ms,49ms,75ms,49ms,48ms,49ms,48ms,50ms,49ms,50ms,49ms,48ms,49ms,50ms,50ms,49ms,48ms,51ms,48ms,
49ms,51ms,48ms,48ms,49ms,49ms,50ms,47ms,51ms,50ms,50ms,49ms,48ms,49ms,49ms,48ms,48ms,48ms,51ms,48ms,48ms,50ms,48ms,50ms,59ms,67ms,50ms,48ms,49ms,49ms,49ms,49ms,
Average: 49

```