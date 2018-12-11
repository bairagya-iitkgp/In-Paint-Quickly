import poissonblending
from config import *
import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from imutils import face_utils
import imutils
import dlib
sys.path.append('..')
if args.res==128:
     if args.model==1:
          from network1 import Network
     if args.model==2:
          from network2 import Network
     if args.model==3:
          from network3 import Network
elif args.res==256:
     if args.model==1:
          from network4 import Network
     if args.model==2:
          from network5 import Network
     if args.model==3:
          from network6 import Network  

IMAGE_SIZE = args.IMAGE_SIZE
LOCAL_SIZE = args.LOCAL_SIZE
HOLE_MIN = args.HOLE_MIN
HOLE_MAX = args.HOLE_MAX
BATCH_SIZE = args.BATCH_SIZE

test_npy = args.test_data_path

detector = dlib.get_frontal_face_detector()


def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 4])
    
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, points, local_x, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, args.restoration_path)

    x_test = np.load(test_npy)
    np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0

    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        if args.res==128:
            points_batch, mask_batch = get_points(x_batch)
        elif args.res==256:
            points_batch, mask_batch = get_points2(x_batch)
       
        completion, imitation = sess.run([model.completion, model.imitation], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw2 = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = raw2 * (1 - mask_batch[i]) 
            mask1 = (mask_batch[i])*255
            mask2 = mask_batch[i]
            img = completion[i]
            img3 = imitation[i]
            img2 = np.array((img + 1) * 127.5, dtype=np.uint8)
            raw2 = cv2.cvtColor(raw2, cv2.COLOR_BGR2RGB)
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
            pm = poisson_blend(raw, img3, 1-mask2)
            pm2 = np.array((pm + 1) * 127.5, dtype=np.uint8)
            pm2 = cv2.cvtColor(pm2, cv2.COLOR_BGR2RGB)
            dst = args.test_out.format("{0:06d}".format(cnt))
            output_image([['P_Blending', pm2]], dst, cnt)
            output_image([['Input', masked], ['Output', img2], ['Ground Truth', raw2], ['Mask Batch', mask1]], dst, cnt)



def get_points(x_batch):
    points = []
    mask = []
    global x,y,w,h
    x,y,w,h=np.array([0,0,0,0])
    for i in range(BATCH_SIZE):
        image = np.array((x_batch[i] + 1) * 127.5, dtype=np.uint8)
        rects = detector(image,1)
        for (k, rect) in enumerate(rects):
            x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
        
        Face_Hole_max_w = min(HOLE_MAX,int(w))
        Face_Hole_max_h = min(HOLE_MAX,int(h))
        L = np.random.randint(HOLE_MIN, Face_Hole_max_w)
        M = np.random.randint(HOLE_MIN, Face_Hole_max_h)
        p1 = x + np.random.randint(0, w - L)
        q1 = y + np.random.randint(0, h - M)
        p2 = p1 + L
        q2 = q1 + M
        p3 = p1 + int(L/2)
        q3 = q1 + int(M/2)
        x1 = p3 - LOCAL_SIZE/2
        if(x1<1):
            t = 1 - x1;
            x1 = 1;
            p2 = p2 + t;
            p1 = p1 + t;
            p3 = p3 + t;
        y1 = q3 - LOCAL_SIZE/2
        if(y1<1):
            t = 1 - y1;
            y1 = 1;
            q2 = q2 + t;
            q1 = q1 + t;
            q3 = q3 + t;
        x2 = p3 + LOCAL_SIZE/2
        if(x2>127):
            t = x2 -127;
            x2 = 127;
            p2 = p2 - t;
            p1 = p1 - t;
            x1 = x1 - t;
        y2 = q3 + LOCAL_SIZE/2
        if(y2>127):
            t = y2 -127;
            y2 = 127;
            q2 = q2 - t;
            q1 = q1 - t;
            y1 = y1 - t;
        points.append([x1, y1, x2, y2])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)



def get_points2(x_batch):
    points = []
    mask = []
    global x,y,w,h
    x,y,w,h=0,0,0,0
    for i in range(BATCH_SIZE):
        image = np.array((x_batch[i] + 1) * 127.5, dtype=np.uint8)
        rects = detector(image,1)
        for (k, rect) in enumerate(rects):
            x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
        
        Face_Hole_max_w = min(HOLE_MAX,int(w))
        
        if(Face_Hole_max_w <= HOLE_MIN):
           Face_Hole_max_w = HOLE_MIN+1 
        
        Face_Hole_max_h = min(HOLE_MAX,int(h))
        
        if(Face_Hole_max_h <= HOLE_MIN):
           Face_Hole_max_h = HOLE_MIN+1
        L = np.random.randint(HOLE_MIN, Face_Hole_max_w)
        M = np.random.randint(HOLE_MIN, Face_Hole_max_h)

        if(w<L):
           p1 = x
        else:
           p1 = x + np.random.randint(0, w - L)

        if(h<M):
           q1 = y
        else:
           q1 = y + np.random.randint(0, h - M)
        p2 = p1 + L
        q2 = q1 + M
        p3 = p1 + int(L/2)
        q3 = q1 + int(M/2)
        x1 = p3 - LOCAL_SIZE/2
        if(x1<1):
            t = 1 - x1;
            x1 = 1;
            p2 = p2 + t;
            p1 = p1 + t;
            p3 = p3 + t;
        y1 = q3 - LOCAL_SIZE/2
        if(y1<1):
            t = 1 - y1;
            y1 = 1;
            q2 = q2 + t;
            q1 = q1 + t;
            q3 = q3 + t;
        x2 = p3 + LOCAL_SIZE/2
        if(x2>255):
            t = x2 -255;
            x2 = 255;
            p2 = p2 - t;
            p1 = p1 - t;
            x1 = x1 - t;
        y2 = q3 + LOCAL_SIZE/2
        if(y2>255):
            t = y2 -255;
            y2 = 255;
            q2 = q2 - t;
            q1 = q1 - t;
            y1 = y1 - t;
        points.append([x1, y1, x2, y2])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)    

def poisson_blend(imgs1, imgs2, mask):
     out = np.zeros(imgs1.shape)
     img1 = (imgs1 + 1.)/2.
     img2 = (imgs2 + 1.)/2.
     out =  np.clip((poissonblending.blend(img1, img2, 1 - mask) - 0.5) * 2, -1.0, 1.0)
     return out

def output_image(images, dst, cnt):
    for i, image in enumerate(images):
        text, img = image
        cv2.imwrite(str(dst)+'img_'+str(cnt)+'_'+text+'.jpg', (img))


if __name__ == '__main__':
    test()
    
