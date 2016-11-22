#-*- coding:utf-8 -*-
#onlyzs1023@gmail.com 2016/11/21
#automatically get images from google image search
import urllib2
import httplib2
import json 
import os

API_KEY = ""
CUSTOM_SEARCH_ENGINE = ""

def getImageUrl(search_item, total_num):
 img_list = []
 i = 0
 while i < total_num:
  query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + str(10 if(total_num-i)>10 else (total_num-i)) + "&start=" + str(i+1) + "&q=" + search_item + "&searchType=image"
  print (query_img)
  res = urllib2.urlopen(query_img)
  data = json.load(res)
  print(len(data["items"]))
  for j in range(len(data["items"])):
   img_list.append(data["items"][j]["link"])
  i=i+10
 return img_list
 
def getImage(img_list):
 opener = urllib2.build_opener()
 http = httplib2.Http(".cache")
 for i in range(len(img_list)):
  try:
   fn, ext = os.path.splitext(img_list[i])
   print(fn, ext)
   response, content = http.request(img_list[i])
   print(response.status)
   with open(str(i)+ext, 'wb') as f:
	f.write(content)
  except:
   print("failed to download images.")
   continue

if __name__ == "__main__":
 img_list = getImageUrl("犬", 15)
 print(img_list)
 getImage(img_list)
