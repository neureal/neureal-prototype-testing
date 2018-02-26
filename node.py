from socket import *
from threading import Thread
import json
import mxnet_model

def use_model():
    # demo connection to mxnet_model module, which holds all our MXNet capabilities.
    print(  mxnet_model.ImagenetModel(  'synset.txt',
                                        'squeezenet_v1.1',
                                        params_url='http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-0000.params',
                                        symbol_url='http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json',
                                        synset_url='http://data.mxnet.io/models/imagenet/synset.txt'
                                    ).predict_from_file('cat.jpg')  )

host = 'localhost'
port = 1234
s = socket(AF_INET, SOCK_STREAM)
s.connect((host, port))

def accept_data(data):
    # TO DO:
    # if data is weights:
    #   sends weights to model ()
    # if data is request of weights:
    #   get weights of model ()
    # else: demo:
    if 'model' in data:
        use_model()
    elif 'shutdown' in data:
        exit()


def listener():
    try:
        while True:
            data = s.recv(1024).decode('utf-8')
            print('incoming message:', data)
            accept_data(data)
    except ConnectionAbortedError:
        pass


t = Thread(target=listener)
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
