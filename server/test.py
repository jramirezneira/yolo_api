import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the MTCNN module for face detection and the InceptionResnetV1 module for face embedding.
mtcnn = MTCNN(image_size=160, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding_and_face(image_path):
    """Load an image, detect the face, and return the embedding and face."""
    try:
        response = requests.get(image_path)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type')
        if 'image' not in content_type:
            raise ValueError(f"URL does not point to an image: {content_type}")
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None, None

    faces, probs = mtcnn(image, return_prob=True)
    if faces is None or len(faces) == 0:
        return None, None

    embedding = resnet(faces[0].unsqueeze(0))
    return embedding, faces[0]


# def tensor_to_image(tensor):
#     """Convert a normalized tensor to a valid image array."""
#     image = tensor.permute(1, 2, 0).detach().numpy()
#     image = (image - image.min()) / (image.max() - image.min())
#     image = (image * 255).astype('uint8')
#     return image

def find_most_similar(target_image_path, candidate_image_paths):
    """Find the most similar image to the target image from a list of candidate images."""
    target_emb, target_face = get_embedding_and_face(target_image_path)
    if target_emb is None:
        raise ValueError("No face detected in the target image.")

    highest_similarity = float('-inf')
    most_similar_face = None
    most_similar_image_path = None

    candidate_faces = []
    similarities = []

    for candidate_image_path in candidate_image_paths:
        candidate_emb, candidate_face = get_embedding_and_face(candidate_image_path)
        if candidate_emb is None:
            similarities.append(None)
            candidate_faces.append(None)
            continue

        similarity = torch.nn.functional.cosine_similarity(target_emb, candidate_emb).item()
        similarities.append(similarity)
        candidate_faces.append(candidate_face)

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_face = candidate_face
            most_similar_image_path = candidate_image_path

    # # Visualization
    # plt.figure(figsize=(12, 8))

    # # Display target image
    # plt.subplot(2, len(candidate_image_paths) + 1, 1)
    # plt.imshow(tensor_to_image(target_face))
    # plt.title("Target Image")
    # plt.axis("off")

    # Display most similar image
    # if most_similar_face is not None:
    #     plt.subplot(2, len(candidate_image_paths) + 1, 2)
    #     plt.imshow(tensor_to_image(most_similar_face))
    #     plt.title("Most Similar")
    #     plt.axis("off")

    # Display all candidate images with similarity scores
    # for idx, (candidate_face, similarity) in enumerate(zip(candidate_faces, similarities)):
    #     plt.subplot(2, len(candidate_image_paths) + 1, idx + len(candidate_image_paths) + 2)
    #     if candidate_face is not None:
    #         plt.imshow(tensor_to_image(candidate_face))
    #         plt.title(f"Score: {similarity * 100:.2f}%")
    #     else:
    #         plt.title("No Face")
    #     plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    if most_similar_image_path is None:
        raise ValueError("No faces detected in the candidate images.")

    return most_similar_image_path, highest_similarity


image_url_target = 'https://d1mnxluw9mpf9w.cloudfront.net/media/7588/4x3/1200.jpg'
candidate_image_urls = [
    'https://beyondthesinglestory.wordpress.com/wp-content/uploads/2021/04/elon_musk_royal_society_crop1.jpg',
    'https://cdn.britannica.com/56/199056-050-CCC44482/Jeff-Bezos-2017.jpg',
    'https://cdn.britannica.com/45/188745-050-7B822E21/Richard-Branson-2003.jpg',
    'https://cdn.britannica.com/70/118070-004-ACF4A6EC/Jeff-Bezos-2005.jpg?w=385',
    'https://ntvb.tmsimg.com/assets/assets/487130_v9_bb.jpg'
]

most_similar_image, similarity_score = find_most_similar(image_url_target, candidate_image_urls)
print(f"The most similar image is: {most_similar_image}")
print(f"Similarity score: {similarity_score * 100:.2f}%")


# from ultralytics import FastSAM

# # Create a FastSAM model
# model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# # Track with a FastSAM model on a video
# results = model.track(source="https://www.youtube.com/watch?v=yHBXsm2H8gs", imgsz=640)


# # import required libraries
# import cv2
# from PIL import Image
# import os
# from vidgear.gears import CamGear
# from vidgear.gears import WriteGear
# from urllib.parse import urlparse
# from utils.general import image_resize

# # open any valid video stream(for e.g `foo.mp4` file)
# dir_path = os.path.dirname(os.path.realpath(__file__))

# # initialize WebGear_RTC app
# # dir_path+"\\test.mp4"
# source="https://www.youtube.com/watch?v=yHBXsm2H8gs"
# if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
#     SourceType="yt"      
# else:
#     SourceType="rtsp"        

# options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720 }
# stream = CamGear(source=source,  stream_mode=True if SourceType=="yt" else False,  logging=True, **options if SourceType=="yt" else {}).start()    


# # define required FFmpeg parameters for your writer
# output_params = {"-f": "rtsp", "-rtsp_transport": "tcp"}

# # Define writer with defined parameters and RTSP address
# # [WARNING] Change your RTSP address `rtsp://localhost:8554/mystream` with yours!
# # writer = WriteGear(
# #     output="rtsp://0.0.0.0:8554/mystream", logging=True, **output_params
# # )
# fps = 30
# w = 1280
# h = 720
# vid_writer = cv2.VideoWriter(
#     "save_pah.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
# )

# # loop over
# while True:

#     # read frames from stream
#     frame = stream.read()
#     # check for frame if Nonetype
#     if frame is None:
#         break
   
    
#     frame = image_resize(frame, height = 720)
#     # cv2.imshow("RTSP View", frame)
    

#     vid_writer.write(frame)


# # safely close video stream
# stream.stop()

# # safely close writer
# vid_writer.release()

# # # from inference import InferencePipeline
# # # from inference.core.interfaces.stream.sinks import render_boxes

# # # pipeline = InferencePipeline.init(
# # # #   model_id="vehicle-detection-3mmwj/1",
# # #   max_fps=0.5,
# # #   confidence=0.3,
# # #   model_id="yolov8n-1280",
# # #     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
# # #   on_prediction=render_boxes,
# # #   api_key="kHurTw6HmW3eMEduzH5b"
# # # )

# # # pipeline.start()
# # # pipeline.join()

# # from inference import InferencePipeline
# # from inference.core.interfaces.camera.entities import VideoFrame

# # # import opencv to display our annotated images
# # import cv2
# # # import supervision to help visualize our predictions
# # import supervision as sv

# # # create a bounding box annotator and label annotator to use in our custom sink
# # label_annotator = sv.LabelAnnotator()
# # box_annotator = sv.BoundingBoxAnnotator()

# # def my_custom_sink(predictions: dict, video_frame: VideoFrame):
# #     # get the text labels for each prediction
# #     labels = [p["class"] for p in predictions["predictions"]]
# #     # load our predictions into the Supervision Detections api
# #     detections = sv.Detections.from_inference(predictions)
# #     # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
# #     image = label_annotator.annotate(
# #         scene=video_frame.image.copy(), detections=detections, labels=labels
# #     )
# #     image = box_annotator.annotate(image, detections=detections)
# #     # display the annotated image
# #     cv2.imshow("Predictions", image)
# #     cv2.waitKey(1)

# # pipeline = InferencePipeline.init(
# #     model_id="yolov8x-1280",
# #     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
# #     on_prediction=my_custom_sink,
# #     api_key="kHurTw6HmW3eMEduzH5b",
# # )

# # pipeline.start()
# # pipeline.join()