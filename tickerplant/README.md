# Tickerplant

### Basic Idea

We have some real-time data sources which are constantly generating data and we want to capture these data and then we:
  1. Save them to a KDB database;
  2. Let other programs know that some new data have arrived, so that these programs can further handle these data;

### Architecture