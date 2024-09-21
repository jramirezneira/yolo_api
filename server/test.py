# import required libraries
import cv2
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import os
from ultralytics import YOLO
from ultralytics.utils.ops import LOGGER, Profile, non_max_suppression, process_mask_native, process_mask, masks2segments, scale_coords, scale_boxes
from ultralytics.utils.plotting import Annotator, colors



model = YOLO("yolov8n-seg.pt")  # load an official model
# open any valid video stream(for e.g `foo.mp4` file)

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720}       
stream = CamGear(source="https://www.youtube.com/watch?v=PtChZ0D7tkE",  stream_mode=True,  logging=True, **options).start()



# loop over
while True:

    # read frames from stream
    im0 = stream.read()

    # check for frame if Nonetype
    if im0 is None:
        break

#     im0=visualize_results_usual_yolo_inference(
#     im0,
#     model,
#     conf=0.4,
#     iou=0.7,
#     segment=False,
#     thickness=5,
#     font_scale=1.0,
#     return_image_array=True
# )
    cv2.imshow("instance-segmentation", im0)

    # {do something with the frame here}

    # write frame to writer
    # annotator = Annotator(im0, line_width=2)
    # results = model(im0)  # predict on an image

    # if results[0].masks is not None:
    #     clss = results[0].boxes.cls.cpu().tolist()
    #     masks = results[0].masks.xy
    #     for mask, cls in zip(masks, clss):
    #         annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True), det_label=model.model.names[int(cls)])

    
    # cv2.imshow("instance-segmentation", im0)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# safely close video stream
stream.stop()

