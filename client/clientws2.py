# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# various performance tweaks and enable live broadcasting
options = {
    "frame_size_reduction": 25,
    #  "enable_live_broadcast": True,
}

# initialize WebGear_RTC app
web = WebGear_RTC(source="rtsp://192.168.1.159:554/11", logging=True, **options)

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=5000)

# close app safely
web.shutdown()