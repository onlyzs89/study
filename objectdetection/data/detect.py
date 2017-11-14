import numpy as np
import os,sys
import tensorflow as tf
import cv2 as cv

sys.path.append('../')

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util


PATH_TO_CKPT = "save/frozen_inference_graph.pb"
PATH_TO_LABELS = "label_map.pbtxt"
CLASS_NUM = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=CLASS_NUM, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    cap = cv.VideoCapture(0)
    
    counter = 1
    while True:
      _, frame = cap.read()
      image_np_expanded = np.expand_dims(frame, axis=0)
      
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
          
      vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=.9)

      cv.imshow("detections", frame)
      cv.imwrite("cap"+str(counter)+".jpg", frame)
      counter+=1
      
      if cv.waitKey(1) >= 0:
        break
