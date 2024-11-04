# import required libraries
import cv2
from PIL import Image
import os
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
from urllib.parse import urlparse
from utils.general import image_resize

# open any valid video stream(for e.g `foo.mp4` file)
dir_path = os.path.dirname(os.path.realpath(__file__))

# initialize WebGear_RTC app
# dir_path+"\\test.mp4"
source="https://www.youtube.com/watch?v=yHBXsm2H8gs"
if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
    SourceType="yt"      
else:
    SourceType="rtsp"        

options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720 }
stream = CamGear(source=source,  stream_mode=True if SourceType=="yt" else False,  logging=True, **options if SourceType=="yt" else {}).start()    


# define required FFmpeg parameters for your writer
output_params = {"-f": "rtsp", "-rtsp_transport": "tcp"}

# Define writer with defined parameters and RTSP address
# [WARNING] Change your RTSP address `rtsp://localhost:8554/mystream` with yours!
# writer = WriteGear(
#     output="rtsp://0.0.0.0:8554/mystream", logging=True, **output_params
# )
fps = 30
w = 1280
h = 720
vid_writer = cv2.VideoWriter(
    "save_pah.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# loop over
while True:

    # read frames from stream
    frame = stream.read()
    # check for frame if Nonetype
    if frame is None:
        break
   
    
    frame = image_resize(frame, height = 720)
    # cv2.imshow("RTSP View", frame)
    

    vid_writer.write(frame)


# safely close video stream
stream.stop()

# safely close writer
vid_writer.release()

# # from inference import InferencePipeline
# # from inference.core.interfaces.stream.sinks import render_boxes

# # pipeline = InferencePipeline.init(
# # #   model_id="vehicle-detection-3mmwj/1",
# #   max_fps=0.5,
# #   confidence=0.3,
# #   model_id="yolov8n-1280",
# #     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
# #   on_prediction=render_boxes,
# #   api_key="kHurTw6HmW3eMEduzH5b"
# # )

# # pipeline.start()
# # pipeline.join()

# from inference import InferencePipeline
# from inference.core.interfaces.camera.entities import VideoFrame

# # import opencv to display our annotated images
# import cv2
# # import supervision to help visualize our predictions
# import supervision as sv

# # create a bounding box annotator and label annotator to use in our custom sink
# label_annotator = sv.LabelAnnotator()
# box_annotator = sv.BoundingBoxAnnotator()

# def my_custom_sink(predictions: dict, video_frame: VideoFrame):
#     # get the text labels for each prediction
#     labels = [p["class"] for p in predictions["predictions"]]
#     # load our predictions into the Supervision Detections api
#     detections = sv.Detections.from_inference(predictions)
#     # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
#     image = label_annotator.annotate(
#         scene=video_frame.image.copy(), detections=detections, labels=labels
#     )
#     image = box_annotator.annotate(image, detections=detections)
#     # display the annotated image
#     cv2.imshow("Predictions", image)
#     cv2.waitKey(1)

# pipeline = InferencePipeline.init(
#     model_id="yolov8x-1280",
#     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
#     on_prediction=my_custom_sink,
#     api_key="kHurTw6HmW3eMEduzH5b",
# )

# pipeline.start()
# pipeline.join()