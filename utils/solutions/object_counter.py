# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
import math

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from utils.solutions.plotting import Annotator, colors
from facenet_pytorch import MTCNN, InceptionResnetV1
check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon
from collections import  Counter, OrderedDict
import traceback


    
def extract_and_process_face_detaction(tracks, frame):
    detected_objects = []
    if hasattr(tracks, 'boxes') and hasattr(tracks, 'names'):
        for box in tracks.boxes.xyxy:
            object_id = int(box[-1])
            object_name = tracks.names.get(object_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            object_image =  frame[int(y1):int(y2), int(x1):int(x2)]

         


            # detected_objects.append((object_name, object_image (x1, y1, x2, y2)))

class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = False

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.counting_list_by_class_in = []
        self.counting_list_by_class_out = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        # self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        reg_counts,
        count_reg_color=(255, 0, 255),
        line_thickness=1,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=False,
        draw_tracks=False,
        count_txt_thickness=1,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2 or 4")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.reg_counts = reg_counts


        

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

   

    def extract_and_process_tracks(self, tracks, index):
        """Extracts and processes tracks for object counting in a video stream."""
        
        if len(self.reg_pts) == 4:
            self.counting_list_by_class_in.clear()
            self.counting_list_by_class_out.clear()
        else:
            self.region_thickness=3
         # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        # Draw region     
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)  
        
        
        if tracks is not None and tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()          

           

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line, color=self.track_color, track_thickness=self.track_thickness
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects
                if len(self.reg_pts) == 4:
                    # if self.counting_region.contains(Point(track_line[-1])):
                    #     if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                    #         self.counting_list_by_class_in.append([track_id, cls])                                             
                    #     else:
                    #         self.counting_list_by_class_out.append([track_id, cls])
                        
                    #     box_label_reverse=True                    
                       
                    if (
                        prev_position is not None
                        and self.counting_region.contains(Point(track_line[-1]))
                        and track_id not in self.counting_list
                    ):
                        self.counting_list.append(track_id)
                        # if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        #     self.counting_list_by_class_in.append([track_id, cls])                                             
                        # else:
                        #     self.counting_list_by_class_out.append([track_id, cls])
                        
                        # self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))
                        # if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        #     self.in_counts += 1                                                
                        # else:
                        #     self.out_counts += 1
                            

                elif len(self.reg_pts) == 2:                    
                    if prev_position is not None:

                        
                        
                        # if track_id  in [x[0] for x in self.counting_list] and track_id not in [x[0] for x in self.counting_list_by_class_in + self.counting_list_by_class_out]:    
                        if track_id  in self.counting_list and track_id not in [x[0] for x in self.counting_list_by_class_in + self.counting_list_by_class_out]:                            
                            # track= next(filter(lambda x: x[0]== track_id, self.counting_list), None)  #[i for i in self.counting_list if i[0]==track_id]
                            cX1 = track_line[-1][0]
                            cY1 = track_line[-1][1]
                            # distance = math.hypot(track[1][0]-cX1, track[1][1]-cY1)   # Point(track_line[-1]).distance(track[1])
                            distance = Point(track_line[-1]).distance(self.counting_region)
                            if distance > 20:
                                aX= self.counting_region.bounds[0]
                                aY= self.counting_region.bounds[1]
                                bX= self.counting_region.bounds[2]
                                bY= self.counting_region.bounds[3]
                                val0 = ((bX - aX)*(cY1 - aY) - (bY - aY)*(cX1 - aX))
                                thresh = 1e-9
                                if val0 >= thresh:
                                    self.in_counts += 1 
                                    self.counting_list_by_class_in.append([track_id, cls]) 
                                else:
                                    self.out_counts += 1
                                    self.counting_list_by_class_out.append([track_id, cls])
                                   


                        distance = Point(track_line[-1]).distance(self.counting_region)
                        # if distance < self.line_dist_thresh and track_id not in [x[0] for x in self.counting_list]:
                            # self.counting_list.append([track_id,track_line[-1]])    
                        if distance < self.line_dist_thresh and track_id not in self.counting_list:
                            self.counting_list.append(track_id)                        
                               
                # Draw bounding box
                if index ==0:
                    self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))
               
          
            # incount_label = f"In Count : {self.in_counts}"
            # outcount_label = f"OutCount : {self.out_counts}"
        # for i in self.counting_list_by_class:
        #     self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))
   
        counts_label= []
        statsIn= []
        statsOut= []

              
        try:
            if self.counting_list_by_class_in:
                statsIn = Counter(list(list(zip(*self.counting_list_by_class_in))[1]))  
                # statsIn=OrderedDict(statsIn.most_common())    
                # counts_label_in.append(f"In")
                # for stat in stats:
                #     counts_label_in.append(f"{self.names[stat]}: {str(stats[stat])}")

            if self.counting_list_by_class_out:
                statsOut = Counter(list(list(zip(*self.counting_list_by_class_out))[1]))  
                # statsOut=OrderedDict(statsOut.most_common())    
                # counts_label_out.append(f"Out")
                # for stat in stats:
                #     counts_label.append(f"{self.names[stat]}: {str(stat****************************************s[stat])}")
           
            counts_label.append(["", "out", "in"])
            for name in self.names:
                    countIn=statsIn[name] if name in [x for x in statsIn] else "-"
                    countOut=statsOut[name] if name in [x for x in statsOut] else "-"

                    if countIn !="-" or countOut !="-":
                        counts_label.append([self.names[name], str(countIn), str(countOut)])


           
        except Exception as e:      
            traceback.print_exception(type(e), e, e.__traceback__)
              
            

        # Display counts based on user choice  
        # if not self.view_in_counts and not self.view_out_counts:
        #     counts_label = None
        # elif not self.view_in_counts:
        #     counts_label = outcount_label
        # elif not self.view_out_counts:
        #     counts_label = incount_label
        # else:
        #    counts_label = f"{incount_label} {outcount_label}"

        if counts_label:
            self.annotator.count_labels(
                count=counts_label,
                # countsOut=counts_label_out,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
                reg_counts=self.reg_counts
            )

    def get_line(x1, y1, x2, y2):
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points
    


    def display_frames(self):
        """Display frame."""
        if self.env_check:
            cv2.namedWindow("Ultralytics YOLOv8 Object Counter")
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(
                    "Ultralytics YOLOv8 Object Counter", self.mouse_event_for_region, {"region_points": self.reg_pts}
                )
            cv2.imshow("Ultralytics YOLOv8 Object Counter", self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        
    def drawBoxes(self, im0, box, label, extra_label, color=(255, 0, 0), line_thickness=1, txt_color=(255, 255, 255)):
        # self.im0=im0
        # self.annotator = Annotator(self.im0,line_thickness)
        # self.annotator.box_label(box, label=class_name, color=color)
        x, y, bw, bh= (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        p1 = (x, y)
        p2 = (x+bw, y+bh)
        cv2.rectangle(im0, p1, p2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            w_text, h_text = cv2.getTextSize(label, 0, fontScale=0.5, thickness=line_thickness)[0]  # text width, height
            outside = p1[1] - h_text >= 3
            p2 = p1[0] + w_text, p1[1] - h_text - 3 if outside else p1[1] + h_text + 3
            cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im0, label, (p1[0], p1[1] - 2 if outside else p1[1] + h_text + 2), 0, 0.5, txt_color, thickness=line_thickness,
                lineType=cv2.LINE_AA,
             )
            
        p1=(x, y+bh)
        p2=(x, y+bh)
        h=0
        for lbl in extra_label: 
            p1=(x, p1[1]+h)
            w, h = cv2.getTextSize(lbl, 0, fontScale=0.5, thickness=line_thickness)[0] 
            p2=(p2[0], p2[1] + h + 3)         
            cv2.rectangle(im0, p1, (p2[0]+w, p2[1]+5), color, -1, cv2.LINE_AA)
            cv2.putText(im0, lbl, (p1[0], p1[1] +h+2), 0, 0.5, txt_color, thickness=line_thickness,
                lineType=cv2.LINE_AA,
             )
            h=h+5
            
        # if label2:
        #     w, h = cv2.getTextSize(label2, 0, fontScale=0.5, thickness=line_thickness)[0]
        #     p1=(x, y+bh)
        #     p2 = x+w,  y+bh + h + 3 
        #     cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)
        #     cv2.putText(im0, label2, (p1[0], p1[1] +h+2), 0, 0.5, txt_color, thickness=line_thickness,
        #         lineType=cv2.LINE_AA,
        #      )
        
        # if label3:
        #     w, h = cv2.getTextSize(label3, 0, fontScale=0.5, thickness=line_thickness)[0]
        #     p5=(x,p3[1]+h+5)
        #     p6 = x+w, p4[1]+ h + 3 
        #     cv2.rectangle(im0, p5, p6, color, -1, cv2.LINE_AA)
        #     cv2.putText(im0, label3, (p5[0], p5[1] +h+2), 0, 0.5, txt_color, thickness=line_thickness,
        #         lineType=cv2.LINE_AA,
        #      )
        # if label4:
        #     w, h = cv2.getTextSize(label4, 0, fontScale=0.5, thickness=line_thickness)[0]
        #     p7=(x,p5[1]+h+5)
        #     p8 = x+w, p6[1]+ h + 3 +3
        #     cv2.rectangle(im0, p7, p8, color, -1, cv2.LINE_AA)
        #     cv2.putText(im0, label4, (p7[0], p7[1] +h+2), 0, 0.5, txt_color, thickness=line_thickness,
        #         lineType=cv2.LINE_AA,
        #      )
            
        return im0

    def start_counting(self, im0, tracks, index):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image

        # if tracks is None:
        #     return im0

        if tracks is not None:
            if tracks[0].boxes.id is None:
                if self.view_img:
                    self.display_frames()
                self.extract_and_process_tracks(None, index)
        # if tracks is None:
        self.extract_and_process_tracks(tracks, index)

        if self.view_img:
            self.display_frames()
        return self.im0#, self.counting_list_by_class
    
   


if __name__ == "__main__":
    ObjectCounter()
