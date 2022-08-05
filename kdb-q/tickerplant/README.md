# Tickerplant

## Basic Idea

We have some real-time data sources which are constantly generating data and we want to capture these data and then we:
  1. Save them to a KDB database;
  2. Let other programs know that some new data have arrived, so that these programs can further handle these data;

One simpler way is polling: we simply use a program to check KDB database on a regular basis (say every 5 seconds)--but
it does not appear to be too proper and may need much more resources to do the same job. According to the [documentation 
of KDB+/q](https://code.kx.com/q/learn/startingkdb/tick/), a tickerplant is the "correct" way to do this.

## Architecture

[<img src="https://github.com/alex-lt-kong/kdb-q-basics/blob/main/tickerplant/architecture.png?raw=true">](https://github.com/alex-lt-kong/kdb-q-basics/blob/main/tickerplant/architecture.png?raw=true)