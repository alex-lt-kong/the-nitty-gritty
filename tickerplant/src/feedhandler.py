from qpython import qconnection
import pandas as pd
import time
import random

q = qconnection.QConnection(host="localhost", port=9526, username='', password='')
# connect to a q process, so that we can execute q statements in Python
q.open()
print(q)
print(f'IPC version: {q.protocol_version}. Is connected: {q.is_connected()}')

q("h:neg hopen `:localhost:9527;")
tickers = ["HSBC", "IBM", "BABA"]
count = 0
while True:
  print(q.sendSync(
    """{[bidPrice;askPrice]
        h(`upd;`quote;(.z.N;`HSBC;bidPrice;askPrice));}
    """,
    random.random()*100,
    -random.random()*100
  ))
  time.sleep(1)
  

q.close()