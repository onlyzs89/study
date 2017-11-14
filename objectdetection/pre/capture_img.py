import numpy as np
import cv2
import argparse,os,sys

IMAGE_DIR = "../data/images"
NAME = "test" #ラベル名

inWidth = 500
inHeight = 500

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video", help="Video File") #ビデオから写真をキャプチャする場合はパラメータを追加
  args = parser.parse_args()
  
  if args.video is not None:
    cap = cv2.VideoCapture(args.video)
  else:
    cap = cv2.VideoCapture(0)
  
  counter = 21
  while True:
    _, frame = cap.read()
    if frame is not None:
      frame = cv2.resize(frame,(inWidth,inHeight))
      cv2.imshow("capture", frame)
      if cv2.waitKey(1) & 0xFF == ord("c"):
        img_name = os.path.join(IMAGE_DIR,NAME+"_"+str(counter)+".jpg")
        cv2.imwrite(img_name, frame)
        print("Captured: "+img_name)
        counter+=1
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else:
      sys.exit(0)
