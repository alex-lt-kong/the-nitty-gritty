# Prefetching

- This is a continuation of [Sequential and random memory access](../05_sequential-and-random-memory-access)

- The example is based on [this answer](https://stackoverflow.com/a/31688096/19634193) on StackExchange.

- The general idea is that we call the gcc's `__builtin_prefetch()` function to proactively fetch data to L1 cache, so
  data access will be ~ 100x faster when they are needed.

## Results:

```
Prefetching enabled: 8.108sec
Prefetching disabled: 12.277sec
```

## Thoughts

- It turns out that it will not be easy to come up with a simple example that demonstrates the effect of prefetching
  since we need to almost deterministically predict what will happen next and this determinism can not be discovered by
  compiler/hardware easily.
