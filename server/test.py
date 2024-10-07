# from inference import InferencePipeline
# from inference.core.interfaces.stream.sinks import render_boxes

# pipeline = InferencePipeline.init(
# #   model_id="vehicle-detection-3mmwj/1",
#   max_fps=0.5,
#   confidence=0.3,
#   model_id="yolov8n-1280",
#     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
#   on_prediction=render_boxes,
#   api_key="kHurTw6HmW3eMEduzH5b"
# )

# pipeline.start()
# pipeline.join()

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv

# create a bounding box annotator and label annotator to use in our custom sink
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    model_id="yolov8n-1280",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=my_custom_sink,
    api_key="kHurTw6HmW3eMEduzH5b",
)

pipeline.start()
pipeline.join()