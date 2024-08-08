# import required libraries
import cv2
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import os
# open any valid video stream(for e.g `foo.mp4` file)

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1080, "CAP_PROP_FRAME_HEIGHT":720}       
stream = CamGear(source="https://www.youtube.com/watch?v=MNn9qKG2UFI",  stream_mode=True,  logging=False, **options).start()

# define required FFmpeg parameters for your writer
output_params = {"-f": "rtsp", "-rtsp_transport": "tcp"}

# Define writer with defined parameters
writer = WriteGear(output = 'rtsp://localhost:5454/test', logging =True, **output_params)


# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()