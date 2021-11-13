from qpython import qconnection
import pandas as pd
import time
import random

q = qconnection.QConnection(host="localhost", port=9526, username='', password='')
q.open()
print(q)

q("h:neg hopen `:localhost:9527;")
tickers = ["HSBC", "IBM", "BABA"]
count = 0
while True:
  q("h(`upd;`quote;(.z.N;`HSBC;123.45;543.21));")

q.close()