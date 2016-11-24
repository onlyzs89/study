# originated from http://d.hatena.ne.jp/shi3z/20160309/1457480722
# modified to adjust to python3

import sys,os
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.error import URLError,HTTPError 
import subprocess
import argparse
import random
from PIL import Image
import os.path

def cmd(cmd):
	return subprocess.getoutput(cmd)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',        type=str,   default='images')
parser.add_argument('--num_of_classes',  type=int,   default=1000)
parser.add_argument('--num_of_pics',   type=int,   default=10)

args = parser.parse_args()

dict={}
for line in open('words.txt', 'r'):
	line=line.split()
	dict[line[0]]=line[1]

ids = open('imagenet.synset.obtain_synset_list', 'r').read()
ids = ids.split()
random.shuffle(ids)

cmd("mkdir %s"%args.data_dir)
for i in range(args.num_of_classes):
	id = ids[i].rstrip()
	category = dict[id]
	cnt = 0
	if len(category)>0:
		cmd("mkdir %s/%s"%(args.data_dir,category))
		print(category)
		try:
			urls=urlopen("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id).read()
			urls=urls.split()
			random.shuffle(urls)

			j=0
			while cnt<args.num_of_pics if args.num_of_pics<len(urls) else len(urls):
				url = urls[j]
				j+=1
				if j>=len(urls):
					break
				url = url.decode('utf-8')
				print(url)

				filename = os.path.split(url)[1]
				try:
					output = "%s/%s/%d_%s"%(args.data_dir,category,cnt,filename)
					urlretrieve(url,output)
					try:
						img = Image.open(output)
						size = os.path.getsize(output)
						if size==2051: #flickr Error
							cmd("rm %s"%output)
							cnt-=1							
					except IOError:
						cmd("rm %s"%output)
						cnt-=1
				except HTTPError as e:
					cnt-=1
					print (e.reason)
				except URLError as e:
					cnt-=1
					print (e.reason)
				except IOError as e:
					cnt-=1
					print (e)
				cnt+=1
		except HTTPError as e:
			print (e.reason)
		except URLError as e:
			print (e.reason)
		except IOError as e:
			print (e)
