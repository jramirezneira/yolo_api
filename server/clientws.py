from vidgear.gears import NetGear
import cv2
from flask_sock import Sock
from flask import Flask
import base64
import json
import asyncio
from ultralytics.utils.ops import LOGGER
# import websockets
# import socket

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

app = Flask(__name__)
sock = Sock(app)

@sock.route('/')
def handler(websocket):
    cont=0
    while True:
        # receive data from server
        frame = client.recv()

        # check for frame if None
        if frame is None:
            break
            
        im0 = cv2.imencode('.jpg', frame)[1].tobytes()
        # dict_result=dict()
        # dict_result["img"] =base64.b64encode(im0).decode('utf-8')

        try:
            # websocket.send(json.dumps(dict_result))
            websocket.send(base64.b64encode(im0).decode('utf-8'))
        except Exception as e:
            LOGGER.warning("websocket.send : %s" % e)
            websocket.close()

       
        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # # close output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()
    websocket.close()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)



