#AutoEncoder Sample
import numpy as np
import tensorflow as tf
import random
import pickle
import cv2
from scipy import stats

import reader
import reader2

FILE_NAME = 'prototype_1/data_batch_'
WIDTH = 32
HEIGHT = 32
DEPTH = 3
LOOP = 30000
BATCH_SIZE = 100

# TF
def weight_variable_conv(shape):
    kW = shape[0]
    kH = shape[1]
    outPlane = shape[3]
    n = kW*kH*outPlane
    stddev = tf.sqrt(2.0/n)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable_conv(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def de_conv2d(x, W,output_shape,strides):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def encoder(x):
    W_conv1 = weight_variable([3, 3, DEPTH, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 16, 8])
    b_conv3 = bias_variable([8])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    return h_pool3
    
def decoder(x, batch_size):
    de_W_conv3 = weight_variable([3, 3, 16, 8])
    de_b_conv3 = bias_variable([16])
    de_h_conv3 = tf.nn.relu(de_conv2d(x, de_W_conv3, [batch_size, WIDTH/4, HEIGHT/4, 16], [1,2,2,1]) + de_b_conv3)
    
    de_W_conv2 = weight_variable([3, 3, 32, 16])
    de_b_conv2 = bias_variable([32])
    de_h_conv2 = tf.nn.relu(de_conv2d(de_h_conv3, de_W_conv2, [batch_size, WIDTH/2, HEIGHT/2, 32], [1,2,2,1]) + de_b_conv2)
    
    de_W_conv1 = weight_variable([3, 3, 3, 32])
    de_b_conv1 = bias_variable([3])
    de_h_conv1 = tf.nn.relu(de_conv2d(de_h_conv2, de_W_conv1, [batch_size, WIDTH, HEIGHT, 3], [1,2,2,1]) + de_b_conv1)
    
    return de_h_conv1

def training():
    label_list = []
    image_list = []
    test_label = []
    test_image = []
    for i in range(1,6):
        # read train file
        r = reader2.fileReader(FILE_NAME + str(i))
        label_list_sub, image_list_sub = r.read_file(False)
        label_list.extend(label_list_sub)
        image_list.extend(image_list_sub)
        r2 = reader2.fileReader(FILE_NAME + str(i))
        label_list_sub, image_list_sub = r2.read_file(True)
        test_label.extend(label_list_sub)
        test_image.extend(image_list_sub)
    
    #shuffle
    train_image = []
    train_label = []
    index_shuf = list(range(len(image_list)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        train_image.append(image_list[i])
        train_label.append(label_list[i])
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(LOOP):
        offset = random.randint(0, int(len(train_image)//BATCH_SIZE)-1) * BATCH_SIZE
        train_image_batch = train_image[offset:(offset + BATCH_SIZE)]
        train_label_batch = train_label[offset:(offset + BATCH_SIZE)]
        optimizer.run(feed_dict={x: train_image_batch})
        if i%100==0:
            cost_eval = cost.eval(feed_dict={x: train_image_batch})
            print('cost: ' + str(cost_eval))
            saver.save(sess, "./model.ckpt")
            tf.train.write_graph(sess.graph_def, 'graph/', 'graph.pb', as_text=False)
            tf.train.write_graph(sess.graph_def, 'graph/', 'graph.pbtxt', as_text=True)
    
def testing():
    ckpt = tf.train.get_checkpoint_state('./')
    last_model = ckpt.model_checkpoint_path
    saver.restore(sess, last_model)
    
    meta_file = open(META_FILE, mode="rb")
    label_dict = pickle.load(meta_file, encoding="bytes")
    
    np.set_printoptions(threshold=np.inf)
    
    image_list = []
    image_list2 = []
    for i in range(1,6):
        # read train file
        r = reader2.fileReader(FILE_NAME + str(i))
        label_list_sub, image_list_sub = r.read_file(False)
        image_list.extend(image_list_sub)
        label_list_sub, image_list_sub = r.read_file(True)
        image_list2.extend(image_list_sub)
    
    
    eocoded_eval = encoded.eval(feed_dict={x:image_list})
    res = np.zeros(128)
    for m in eocoded_eval:
        for n in range(0,len(m.flatten())):
            res[n] += m.flatten()[n]
    res = res/len(eocoded_eval)
    
    
    # 検定
    list0 = []
    list8 = []
    counter0 = 0
    counter8 = 0
    for i in range(200):
        res1 = cost.eval(feed_dict={x:[image_list[i]]})
        if res1>0.0047:
            counter0+=1
        res2 = cost.eval(feed_dict={x:[image_list2[i]]})
        if res2>0.0047:
            counter8+=1
        list0.append(res1)
        list8.append(res2)
    
    t, p = stats.ttest_rel(list0, list8)
    print("平均差の検定　p値 = %(p)s" %locals())
    print("\nクロス表：")
    print("\t正常A\t異常A")
    print("正常P\t" + str(200-counter0) + "\t" + str(200-counter8))
    print("異常P\t" + str(counter0) + "\t" + str(counter8))
    _,p2,_,_ = stats.chi2_contingency(np.array([[200-counter0, counter0], [200-counter8, counter8]]))
    
    print("\nカイ二乗検定　p値 = %(p2)s" %locals())
    
# tf model
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, WIDTH*HEIGHT*DEPTH], name='input')
x_true = tf.reshape(x, [-1,WIDTH,HEIGHT,DEPTH])
encoded = encoder(x_true)
#x_pred= decoder(encoded,BATCH_SIZE) #train
x_pred= decoder(encoded,1) #test

cost = tf.reduce_mean(tf.pow(x_true - x_pred, 2))
optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(cost)

# create saver
saver = tf.train.Saver()

testing()
#training()
