import os
import random
import sys
import time

import wolk 

import cv2
import numpy as np

import tensorflow as tf

from utils import label_map_util 
from utils import visualization_utils as vis_util  

import socket
import argparse

from pathlib import Path
HOME = str(Path.home())

parser = argparse.ArgumentParser(description='set the IP address.')
parser.add_argument('--IP',type=str, help='set the IP address of the rPi (server device)')

args = parser.parse_args()
host_0 = str(args.IP) #ip of raspberry pi
port_0 = 12345

print("Initialising client of rPi server..")
s = socket.socket()
try: 
    s.connect((host_0, port_0))
    print("#0 Succesfully initialised client of rPi server")
except NameError:
    print("#0 Unsuccesfully initialised client of rPi server")
    sys.exit(-1)

def receivement_from_rPi_server():
    s.send(b'0')
    drunk_val = bool(s.recv(1024))
    print('received: ', str(drunk_val) )
    return drunk_val

time.sleep(1)


class Human:
    def __init__(self):
        self.image = None
        self.detected = False
        self.coordinates = None

def human_detection(image, cascade = cv2.CascadeClassifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    visualization = image.copy()
    
    if(faces is None):
        return False

    coordinate_list = []
    size = len(faces)
    incr1 = 0
    for (x, y, w, h) in faces:
        #print('width: ' + str(w) + ' ; height: ' + str(h) )
        if(w < 100 or h < 100):
            size -= 1
        else:
            coordinate_list.append([x,y,w,h])

    human = Human()
    human.image = visualization    
    if(size > 0):
        human.detected = True
        coordinate_set = coordinate_list[0]
        human.coordinates = coordinate_set
    return human

# Enable debug logging by uncommenting the following line
# wolk.logging_config("debug", "wolk.log")

def Category_Index(path_to_labels = str, num_classes = int):
    label_map = label_map_util.load_labelmap(path_to_labels) 
    categories = label_map_util.convert_label_map_to_categories( 
        label_map, 
        max_num_classes = num_classes, 
        use_display_name = True
    )
    return label_map_util.create_category_index(categories) 

def Detection_Graph(path_to_ckpt = str):
    detection_graph = tf.Graph()
    with detection_graph.as_default(): 
        od_graph_def = tf.compat.v1.GraphDef() 
        with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid: 
            serialized_graph = fid.read() 
            od_graph_def.ParseFromString(serialized_graph) 
            tf.import_graph_def(od_graph_def, name ='')
    return detection_graph

class Detection_Model:
    def __init__(self, path_to_labels = str, num_classes = int, path_to_ckpt = str):
        self.category_index = Category_Index(path_to_labels, num_classes)
        detection_graph = Detection_Graph(path_to_ckpt)
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config, graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def parse_string(string = str):
    name_assigned = False
    name = ""
    percentage = ""
    for letter in string:
        if(letter != ':' and name_assigned != True):
            name += letter
        elif(letter == ':' and name_assigned != True):
            name_assigned = True
        elif(letter >= '0' and letter <= '9' and name_assigned == True):
            percentage += letter

    if(percentage != ""):
        return [name, int(percentage)]
    return ['/', 0]
        
class Admin:
    def __init__(self):
        self.detected = False
        self.name = '/'
        self.coordinates_set = None
        self.detection_string = str()

def admin_detection(image, det_model = Detection_Model):
    image_expanded = np.expand_dims(image, axis = 0)

    (boxes, scores, classes, num) = det_model.session.run( 
        [det_model.detection_boxes, det_model.detection_scores, det_model.detection_classes, det_model.num_detections], 
        feed_dict ={det_model.image_tensor: image_expanded})

    (image, det_string) = vis_util.visualize_boxes_and_labels_on_image_array( 
        image, 
        np.squeeze(boxes), 
        np.squeeze(classes).astype(np.int32), 
        np.squeeze(scores), 
        det_model.category_index, 
        use_normalized_coordinates = True, 
        line_thickness = 8, 
        min_score_thresh = 0.60
        )

    height, width, channels = image.shape

    ymin = int((boxes[0][0][0]*height))
    xmin = int((boxes[0][0][1]*width))
    ymax = int((boxes[0][0][2]*height))
    xmax = int((boxes[0][0][3]*width))

    (name, percentage) = parse_string(det_string[0])
    
    admin = Admin()
    if(percentage >= 90):
        admin.detected = True
        admin.name = name
        admin.coordinates_set = (xmin, ymin, xmax-xmin, ymax-ymin)
        admin.detection_string = det_string

    return admin

def to_Cloud(info1=bool, info2=bool, info3=str, info4=bool, device=wolk.WolkConnect):
    device.add_sensor_reading("Human", info1)
    device.add_sensor_reading("Admin", info2)
    device.add_sensor_reading("Which Admin", info3)
    device.add_sensor_reading("Drunk", info4)
    device.publish()
    print('Publishing \n\t"Human": ' + str(info1) + '\n\t"Admin": ' + str(info2) + ' -> ' + info3 + '\n\t"Drunk": ' + str(info4) + '\n') 

def increase_rectangle(coordinates, image):
    height, width, channels = image.shape
    x, y, w, h = coordinates
    for i in range(int(h/3.5),0,-1):
        if(x-i >= 0 and y-i >= 0 and x+w+2*i <= width and y+h+2*i <= height):
            x -= i
            y -= i
            w += 2*i
            h += 2*i
            break
    return (x,y,w,h)

def image_roi(image, coordinates):
    ROI = image.copy()
    height, width, channels = image.shape
    (x,y,w,h) = coordinates

    external_poly1 = np.array( [[ [0,0],[width,0],[width,y],[0,y] ]], dtype=np.int32) #  [x,y],[x+w,y],[x+w,y+h],[x,y+h] ROI
    external_poly2 = np.array( [[ [0,y],[x,y],[x,y+h],[0,y+h] ]], dtype=np.int32)
    external_poly3 = np.array( [[ [0,y+h],[width,y+h],[width,height],[0,height] ]], dtype=np.int32)
    external_poly4 = np.array( [[ [x+w,y],[width,y],[width,y+h],[x+w,y+h] ]], dtype=np.int32)

    cv2.fillPoly(ROI, external_poly1, (0,0,0) )
    cv2.fillPoly(ROI, external_poly2, (0,0,0) )
    cv2.fillPoly(ROI, external_poly3, (0,0,0) )
    cv2.fillPoly(ROI, external_poly4, (0,0,0) )
    return ROI

def draw_on_frame(image, coordinates, drawing_string):
    colors = [(255, 0, 0), (0,255,0)]

    if(isinstance(drawing_string, str) == True ):    
        if(drawing_string.startswith("human") ):
            colors = [(0,0,255),(0,0,255)]
            text = 'human'
    else:
        text = drawing_string[0]
    
    (x,y,w,h) = coordinates
    cv2.rectangle(image, (x, y), (x+w, y+h), colors[0], 2)
    cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[1] ) 
    return image

PROJECT_DIR_PATH = HOME + '/IoT_detection_PES/dependencies/'

print('\n\nConnecting to WolkAbout IoT Platform and send camera sensor reading.')
# Insert the device credentials received
# from WolkAbout IoT Platform when creating the device
device = wolk.Device(key="6k0zwp7agnw7v5h6", password="95fdba80-4733-48fb-8bad-3a72a9ff375f")

try:
    wolk_device = wolk.WolkConnect(
        device=device,
        protocol=wolk.Protocol.JSON_SINGLE,
        host="iot-elektronika.ftn.uns.ac.rs",
        port=1883
    )
    print("#1 Wolk Connection successful.")
except RuntimeError as e:
    print("#1 Wolk Connection unsuccessful.")
    print(str(e))
    sys.exit(-1)

time.sleep(1)

# Establish a connection to the WolkAbout IoT Platform
print("\n\nConnecting to WolkAbout IoT Platform")
try:
    wolk_device.connect()
    print("#2 Connection unsuccessful.")
except RuntimeError as e:
    print("#2 Connection unsuccessful.")
    print(str(e))
    sys.exit(-1)

time.sleep(1)

print("\n\nInitialising OpenCV objects")

PATH_TO_CLASSIFIER = PROJECT_DIR_PATH + 'cv_cascade.xml'
face_cascade = cv2.CascadeClassifier(PATH_TO_CLASSIFIER)
if(face_cascade.empty == False):
    print("#3 OpenCV Classifier not found in path:\n" + PATH_TO_CLASSIFIER)
    sys.exit(-1)
else:
    print("#3 OpenCV Classifier succesfully found")

webcam = cv2.VideoCapture(0)
if(webcam.isOpened == False):
    print("#4 OpenCV WebCam not found.")
    sys.exit(-1)
else:
    print("#4 OpenCV WebCam sucessfully found.")

time.sleep(1)

print("\n\nInitialising Tensorflow objects\n")

PATH_TO_CKPT = PROJECT_DIR_PATH + 'tf_model.pb'
PATH_TO_LABELS = PROJECT_DIR_PATH + 'labelmap.pbtxt'
NUM_CLASSES = 2
detection_model = Detection_Model(PATH_TO_LABELS, NUM_CLASSES, PATH_TO_CKPT)

try:
    ret, frame = webcam.read()
    admin = admin_detection(frame, detection_model)
    print('\n\n\n#5 Sucessfully initialiesed Tensorflow object detection API\n\n')
except RuntimeError as e:
    print('\n\n\n#5 Unsucessfully initialiesed Tensorflow object detection API')
    sys.exit(-1)

time.sleep(1)
print('\nProgram started\n')

Time = 0
play = 1

def succ_exit():
    webcam.release()
    cv2.destroyAllWindows()
    s.close()
    exit()


def reset():
    global human_det_report 
    global admin_det_report
    global admin_name_report
    global drunk_report

    human_det_report = False
    admin_det_report = False
    admin_name_report = '/'
    drunk_report = False

reset()

def do_detection(image):
    global human_det_report 
    global admin_det_report
    global admin_name_report
    global drunk_report

    human = human_detection(image, face_cascade)
    human_det_report = human.detected
    coordinates = human.coordinates
    if(human_det_report == True):
        coordinates2 = increase_rectangle(coordinates, image)
        ROI_frame = image_roi(image, coordinates2)
        admin = admin_detection(ROI_frame, detection_model)
        admin_det_report = admin.detected
        admin_name_report = admin.name
        if(admin_det_report == True):
            coordinates = admin.coordinates_set
            drawing_string = admin.detection_string
            print("\nAdmin detected")
        else:
            drawing_string = "human"
            print("\nHuman detected")
        time_1 = time.time()
        print('Press any button to get value from sensor')
        frame = draw_on_frame(image, coordinates, drawing_string )
        cv2.imshow('stream', image)
        cv2.waitKey(0)
        time_2 = time.time()
        drunk_report = receivement_from_rPi_server()
        return time_2-time_1
    else:
        print("\nNo human detected")
        drunk_report = False
        return 0

while True:
    global human_det_report 
    global admin_det_report
    global admin_name_report
    global drunk_report

    time_start = time.time()
    
    ret, frame = webcam.read()

    if(Time == 0):
        diff = do_detection(frame)
        to_Cloud(human_det_report, admin_det_report, admin_name_report, drunk_report, wolk_device)
        reset()

    cv2.imshow('stream', frame)

    btn = cv2.waitKey(1)
    if btn & 0xFF == ord('q'):
        break

    time_end = time.time() 
    Time += (time_end - diff - time_start)
    diff = 0

    if( Time >= 5 ):
        Time = 0

succ_exit()
    
    
