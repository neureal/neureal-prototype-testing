from socket import *
from threading import Thread
import json
import load_model

def use_model():
    print(  load_model.ImagenetModel(   'synset.txt',
                                        'squeezenet_v1.1',
                                        params_url='http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-0000.params',
                                        symbol_url='http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json',
                                        synset_url='http://data.mxnet.io/models/imagenet/synset.txt'
                                    ).predict_from_file('cat.jpg')  )

host = 'localhost'
port = 1234
s = socket(AF_INET, SOCK_STREAM)
s.connect((host, port))

def Listener():
    try:
        while True:
            data = s.recv(1024).decode('utf-8')
            print('incoming message:', data)
            # Here we need to do something based on the data we get (like use_model)
    except ConnectionAbortedError:
        pass


t = Thread(target=Listener)
t.start()

def get_input():
    to = input('to: ')
    msg = input('msg: ')
    return to, msg


try:
    while True:
        to, msg = get_input()
        data = json.dumps({'to':to,'msg':msg})
        s.send(data.encode('utf-8'))
except EOFError:
    pass
finally:
    s.close()
