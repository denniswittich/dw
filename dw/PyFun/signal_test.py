import signal
import sys

run = True
try:
    while(run):
        print('a')
except KeyboardInterrupt:
    print('b')

def signal_handler(signal, frame):
    global run
    run = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)