import numpy as np
import cv2
import glob,os

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

IMAGE_DIR = "..\\data\\images"
ANNOTATION_DIR = "..\\data\\annotations"
NAME = "test"

class BndBox:
  def __init__(self, img, fname):
    self.inHeight, self.inWidth = img.shape[:2]
    self.xmin = 0
    self.ymin = 0
    self.xmax = 0
    self.ymax = 0
    self.drawing = False
    self.img = img
    self.fname = fname
    
  def mouse_event(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.drawing = True
      self.xmin = x
      self.ymin = y
    elif event == cv2.EVENT_MOUSEMOVE:
      if self.drawing:
        img_copy = self.img.copy()
        cv2.rectangle(img_copy,(self.xmin,self.ymin),(x,y),(0,255,0),1)
        cv2.imshow(self.fname, img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
      self.drawing = False
      self.xmax = x
      self.ymax = y
      img_copy = self.img.copy()
      cv2.rectangle(img_copy,(self.xmin,self.ymin),(x,y),(0,255,0),1)
      cv2.imshow(self.fname, img_copy)
      
  def clear(self):
    self.xmin = 0
    self.ymin = 0
    self.xmax = 0
    self.ymax = 0
    cv2.imshow(self.fname, self.img)
    
  def save(self):
    annotation = ET.Element('annotation')
    
    filename = ET.SubElement(annotation, 'filename')
    filename.text = self.fname + ".jpg"
    
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(self.inWidth)
    height = ET.SubElement(size, 'height')
    height.text = str(self.inHeight)
    depth = ET.SubElement(size, 'depth')
    depth.text = "3"
    
    object = ET.SubElement(annotation, 'object')
    pose = ET.SubElement(object, 'pose')
    pose.text = "Unspecified"
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = "0"
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = "0"
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(self.xmin)
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(self.ymin)
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(self.xmax)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(self.ymax)
    
    string = ET.tostring(annotation, 'utf-8')
    pretty_string = minidom.parseString(string).toprettyxml(indent='  ')
    
    xml_file = os.path.join(ANNOTATION_DIR,"xmls",self.fname + '.xml')
    with open(xml_file, 'w') as f:
      f.write(pretty_string)

if __name__ == "__main__":
  files = glob.glob(IMAGE_DIR+"/*.jpg")
  trainval = []
  
  for f in files:
    img = cv2.imread(f)
    
    fname = os.path.splitext(os.path.basename(f))[0]
    bndBox = BndBox(img,fname)
    
    cv2.namedWindow(fname)
    cv2.setMouseCallback(fname, bndBox.mouse_event)
    
    while (True):
      cv2.imshow(fname, img)
      if cv2.waitKey(1) & 0xFF == ord("n"):
        bndBox.save()
        trainval.append(fname+" "+"1")
        print(bndBox.xmin,bndBox.ymin,bndBox.xmax,bndBox.ymax,fname+" saved")
        break
      if cv2.waitKey(1) & 0xFF == ord("c"):
        bndBox.clear()
      
    cv2.destroyAllWindows()
  
  txt_file = os.path.join(ANNOTATION_DIR,"trainval.txt")
  with open(txt_file, "w", encoding="utf-8") as f:
    f.write("\n".join(trainval))
  
