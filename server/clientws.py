# import required libraries
from vidgear.gears import NetGear
import cv2
# from flask_sock import Sock
# from flask import Flask
import base64
import json
import asyncio
 
import websockets

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

async def handler(websocket, path):
    cont=0
    while True:
        # receive data from server
        frame = client.recv()

        # check for frame if None
        if frame is None:
            break
        
       
        # cont=cont+1
            
        im0 = cv2.imencode('.jpg', frame)[1].tobytes()
        dict_result=dict()
        dict_result["img"] =base64.b64encode(im0).decode('utf-8')

        await websocket.send(json.dumps(dict_result))

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

start_server = websockets.serve(handler, "0.0.0.0", 5000)
asyncio.get_event_loop().run_until_complete(start_server)
 
asyncio.get_event_loop().run_forever()
# if __name__ == "__main__":


