# import required libraries
from vidgear.gears.asyncio import WebGear_RTC
import uvicorn
import cv2
import time
import os
 
def openStreamRtspServer(source):
    while True:
        try:
            video=cv2.VideoCapture(source)
            if video is None or not video.isOpened():
                raise ConnectionError
            else:
                web = WebGear_RTC(source=source, logging=True)
                uvicorn.run(web(), host="0.0.0.0", port=5000)
        except ConnectionError:
                print("Retrying connection to ",source," in ",str(1.5), " seconds...")
                time.sleep(1.5)

openStreamRtspServer("rtsp://127.0.0.1:8554/mystream")
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)

# openStreamRtspServer("%s/test.mp4" % dir_path )

