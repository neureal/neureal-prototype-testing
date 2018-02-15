from socket import *
from threading import Thread

host = 'localhost'
port = 1234
s = socket(AF_INET, SOCK_STREAM)
s.connect((host, port))

def Listener():
    try:
        while True:
            data = s.recv(1024).decode('utf-8')
            print('', data)
    except ConnectionAbortedError:
        pass


t = Thread(target=Listener)
t.start()

try:
    while True:
        message = input('')
        s.send(message.encode('utf-8'))
except EOFError:
    pass
finally:
    s.close()
