# import required libraries
import uvicorn
import time
import cv2
from vidgear.gears.asyncio import WebGear_RTC
# from utils.general import image_resize, getConfProperty, setProperty

# various performance tweaks and enable live broadcasting
options = {
    "frame_size_reduction": 5,
      "enable_live_broadcast": True,
      "enable_infinite_frames":True,
      "CAP_PROP_FPS":20
    #   "CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60
}

# rtspServer= "192.168.1.159"#, _ =getConfProperty("rtspServer") 
rtspServer= "192.168.1.167:8554/mystream"
web =None

def openStreamRtspServer():
    while True:
        try:
            # initialize WebGear_RTC app
            return WebGear_RTC(source="rtsp://%s" % rtspServer,stabilize=True,  logging=True,framerate=20,   **options)   
            # return WebGear_RTC(source="https://www.youtube.com/watch?v=mpdN4pZL_vM",stabilize=True, logging=True,framerate=30,  stream_mode=True, **options)   
            
        except  Exception:
                print("Retrying connection to rtsp server, in ",str(1.5), " seconds...")
                time.sleep(1.5)

web=openStreamRtspServer()

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=5000)

# close app safely
web.shutdown()