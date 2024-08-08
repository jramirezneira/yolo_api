# import required libraries
from vidgear.gears import NetGear
import cv2
# from flask_sock import Sock
# from flask import Flask
import base64
import json
import socket
 
# import thread module
from _thread import *
import threading

print_lock = threading.Lock()
 
# import asyncio
 
# import websockets

# activate Multi-Clients mode
options = {"multiclient_mode": True}
client = NetGear(
        address="0.0.0.0",
        port="5567",
        protocol="tcp",
        pattern=2,
        receive_mode=True,
        logging=True,
        **options
    ) 

def handler(c):
    cont=0
    while True:
        # receive data from server
        frame = client.recv()

        # check for frame if None
        if frame is None:
            print_lock.release()
            break
        
       
        cont=cont+1
        print(cont)
            
        im0 = cv2.imencode('.jpg', frame)[1].tobytes()
        dict_result=dict()
        dict_result["img"] =base64.b64encode(im0).decode('utf-8')

        try:
            c.sendall(json.dumps(dict_result).encode('utf-8'))
        except Exception as e:
            print("pong : %s" % e)
        # {do something with frame here}

        # Show output window
        # cv2.imshow("Client 5567 Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # # close output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()
    c.close()


def Main():
    host = "0.0.0.0"
 
    # reserve a port on your computer
    # in our case it is 12345 but it
    # can be anything
    port = 5000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to port", port)
 
    # put the socket into listening mode
    s.listen(5)
    print("socket is listening")
 
    # a forever loop until client wants to exit
    while True:
 
        # establish connection with client
        c, addr = s.accept()
 
        # lock acquired by client
        print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])
 
        # Start a new thread and return its identifier
        start_new_thread(handler, (c,))
    s.close()
 
 
if __name__ == '__main__':
    Main()
# if __name__ == "__main__":


