# import required libraries
import uvicorn
import time
from vidgear.gears.asyncio import WebGear_RTC
from utils.general import image_resize, getConfProperty, setProperty

# various performance tweaks and enable live broadcasting
options = {
    "frame_size_reduction": 25,
    #  "enable_live_broadcast": True,
}

rtspServer, _ =getConfProperty("rtspServer") 
web =None

def openStreamRtspServer():
    while True:
        try:
            # initialize WebGear_RTC app
            return WebGear_RTC(source="rtsp://%s:8554/mystream" % rtspServer, logging=True, **options)   
            
        except  Exception:
                print("Retrying connection to rtsp server, in ",str(1.5), " seconds...")
                time.sleep(1.5)

web=openStreamRtspServer()

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=5000)

# close app safely
web.shutdown()