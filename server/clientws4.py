# import required libraries
import uvicorn
import os
from vidgear.gears.asyncio import WebGear_RTC

# various performance tweaks and enable live broadcasting
options = {
    "frame_size_reduction": 25,
    "enable_live_broadcast": True,
}

dir_path = os.path.dirname(os.path.realpath(__file__))

# initialize WebGear_RTC app
# dir_path+"\\test.mp4"
web = WebGear_RTC(source="rtsp://localhost:8554/mystream", logging=True, **options)

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=8080)

# close app safely
web.shutdown()

# from vidgear.gears import NetGear
# import cv2

# from ultralytics.utils.ops import LOGGER

# # activate Multi-Clients mode
# class Custom_Stream_Class:
#     def __init__(self):   
#         self.running = True
#         self.source=None
#         options = {"multiclient_mode": True}
#         self.client = NetGear(
#                             address="0.0.0.0",
#                             port="5567",
#                             protocol="tcp",
#                             pattern=2,
#                             receive_mode=True,
#                             logging=True,
#                             **options
#                         ) 


#     def read(self):
#         cont=0
#         while True:
#             # receive data from server
#             frame = self.client.recv()

#             # check for frame if None
#             if frame is None:
#                 break
                
#             # im0 = cv2.imencode('.jpg', frame)[1].tobytes()
    
#         # # close output window
#         cv2.destroyAllWindows()

#         # safely close client
#         self.client.close()

#     def close(self):
#         self.client.close()



