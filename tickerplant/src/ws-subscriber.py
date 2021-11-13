import json
import websocket
import _thread
import time

def on_message(ws, message):
  print('hello!')

def on_error(ws, error):
  print(error)

def on_close(ws, close_status_code, close_msg):
  print(f"Connection is closed, status_code: {close_status_code}")

def on_open(ws):
  ws.send("loadPage[]")

if __name__ == "__main__":

  websocket.enableTrace(True)
  ws = websocket.WebSocketApp(
    "ws://localhost:9527",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
  )

  ws.run_forever()

