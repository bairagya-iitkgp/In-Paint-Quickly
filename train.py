import numpy as np
import tensorflow as tf
from config import *
import cv2
import tqdm
if args.res==128:
     if args.model==1:
          from network1 import Network
     if args.model==2:
          from network2 import Network
     if args.model==3:
          from network3 import Network
elif args.res==256:
     if args.model==2:
          from network5 import Network
     if args.model==3:
          from network6 import Network   
import load
import logging
from imutils import face_utils
import imutils
import dlib

IMAGE_SIZE = args.IMAGE_SIZE
LOCAL_SIZE = args.LOCAL_SIZE
HOLE_MIN = args.HOLE_MIN
HOLE_MAX = args.HOLE_MAX
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
PRETRAIN_EPOCH = args.PRETRAIN_EPOCH
Td_EPOCH = args.Td_EPOCH
Tot_EPOCH = args.Tot_EPOCH

logging.basicConfig(level=logging.DEBUG, filename=args.log,filemode='a+',format='%(asctime)s %(message)s')

dir1=args.output
dir2=args.original
dir3=args.perturbed

detector = dlib.get_frontal_face_detector()

def train():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 4])

    model = Network(x, mask, points, local_x, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    g_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    combined_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.combined_loss, global_step=global_step, var_list=model.g_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if args.use_pretrain==True:
        variables_r = tf.trainable_variables()
        variables_to_restore = [var for var in variables_r if 'new_var' not in var.name] 
        model_path = args.pretrain_path
        saver1 = tf.train.Saver(variables_to_restore)
        saver1.restore(sess, model_path)

    if tf.train.get_checkpoint_state(args.checkpoints_path):
        saver = tf.train.Saver()
        saver.restore(sess, args.restoration_path)

    x_train, x_test = load.load()
    x_train = np.array([a / 127.5 - 1 for a in x_train])
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_train) / BATCH_SIZE)

    while sess.run(epoch) < Tot_EPOCH:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        np.random.shuffle(x_train)

        # Completion Network Training
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)

                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                
                if i%50==0:
                    print('Completion loss: {}'.format(g_loss))
                    logging.info('epoch: %s   Completion loss:  %s  ', str(sess.run(epoch)),str(g_loss))
                   
            print('Completion loss: {}'.format(g_loss))
            logging.info('epoch: %s   Completion loss:  %s  ', str(sess.run(epoch)),str(g_loss))
            np.random.shuffle(x_test)
            step_num2 = int(len(x_test) / BATCH_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)
                completion2, c_loss = sess.run([model.completion, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss;

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_SIZE]
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            img_tile(dir1, cnt,gc_loss, completion)
            img_tile2(dir2, cnt,gc_loss, x_batch)
            img_tile2(dir3, cnt,gc_loss, x_pert)
            saver = tf.train.Saver()
            saver.save(sess, args.restoration_path, write_meta_graph=False)


        # Discrimitation Network Training
        elif sess.run(epoch) <= (PRETRAIN_EPOCH+Td_EPOCH):
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)
                               
                local_x_batch = []
             
                for j in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[j]
                    local_x_batch.append(x_batch[j][int(y1):int(y2), int(x1):int(x2), :])
                    
                local_x_batch = np.array(local_x_batch)
                

                _, d_loss, dfake_loss = sess.run(
                    [d_train_op, model.d_loss, model.dfake_loss], 
                    feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})

                if i%50==0:
                    print('Discriminator loss: {}'.format(d_loss))
                    logging.info('epoch: %s Discriminator loss:  %s GAN loss:  %s ', str(sess.run(epoch)), str(d_loss), str(dfake_loss))
            print('Discriminator loss: {}'.format(d_loss))
            logging.info('epoch: %s Discriminator loss:  %s GAN loss:  %s ', str(sess.run(epoch)), str(d_loss), str(dfake_loss)) 
            np.random.shuffle(x_test)
            step_num2 = int(len(x_test) / BATCH_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)
                completion2, c_loss = sess.run([model.completion, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss;

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_SIZE]
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            img_tile(dir1, cnt,gc_loss, completion)
            img_tile2(dir2, cnt,gc_loss, x_batch)
            img_tile2(dir3, cnt,gc_loss, x_pert)
            saver = tf.train.Saver()
            saver.save(sess, args.restoration_path, write_meta_graph=False)
            
        else:
            # Combined Training 
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)
                               
                local_x_batch = []
                
                for j in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[j]
                    local_x_batch.append(x_batch[j][int(y1):int(y2), int(x1):int(x2), :])
                    
                local_x_batch = np.array(local_x_batch)
                
                _, d_loss = sess.run(
                    [d_train_op, model.d_loss], 
                    feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})
                
                _, combined_loss, g_loss, dfake_loss = sess.run([combined_op, model.combined_loss, model.g_loss, model.dfake_loss], feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})

                
                if i%50==0:
                    print('Combined loss: {}'.format(combined_loss))
                    print('Discriminator loss: {}'.format(d_loss))
                    logging.info('epoch: %s combined loss:  %s Discriminator loss:  %s ', str(sess.run(epoch)), str(combined_loss), str(d_loss))
                    logging.info('epoch: %s MSE loss:  %s GAN loss: %s', str(sess.run(epoch)), str(g_loss),str(dfake_loss))
                                        
            print('Combined loss: {}'.format(combined_loss))
            print('Discriminator loss: {}'.format(d_loss))
            logging.info('epoch: %s combined loss:  %s Discriminator loss:  %s ', str(sess.run(epoch)), str(combined_loss), str(d_loss))
            logging.info('epoch: %s MSE loss:  %s GAN loss: %s', str(sess.run(epoch)), str(g_loss),str(dfake_loss))
           
            np.random.shuffle(x_test)
            step_num2 = int(len(x_test) / BATCH_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                if args.res==128:
                     points_batch, mask_batch = get_points(x_batch)
                elif args.res==256:
                     points_batch, mask_batch = get_points2(x_batch)     
                completion2, c_loss = sess.run([model.completion, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss;

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_SIZE]
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            img_tile(dir1, cnt,gc_loss, completion)
            img_tile2(dir2, cnt,gc_loss, x_batch)
            img_tile2(dir3, cnt,gc_loss, x_pert)
            saver = tf.train.Saver()
            saver.save(sess, args.restoration_path, write_meta_graph=False)



#Generating Holes inside Face Region(128X128 Input)
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

#Generating Holes inside Face Region(256X256 Input)
def get_points2(x_batch):
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



#Assembling Images in Tile Format
def img_tile(loc, cnt, out_loss, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
 	if imgs.ndim != 3 and imgs.ndim != 4:
 		raise ValueError('imgs has wrong number of dimensions.')
 	n_imgs = imgs.shape[0]

 	tile_shape = None
 	# Grid shape
 	img_shape = np.array(imgs.shape[1:3])
 	if tile_shape is None:
 		img_aspect_ratio = img_shape[1] / float(img_shape[0])
 		aspect_ratio *= img_aspect_ratio
 		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
 		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
 		grid_shape = np.array((tile_height, tile_width))
 	else:
 		assert len(tile_shape) == 2
 		grid_shape = np.array(tile_shape)

 	# Tile image shape
 	tile_img_shape = np.array(imgs.shape[1:])
 	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

 	# Assemble tile image
 	tile_img = np.empty(tile_img_shape)
 	tile_img[:] = border_color
 	for i in range(grid_shape[0]):
 		for j in range(grid_shape[1]):
 			img_idx = j + i*grid_shape[1]
 			if img_idx >= n_imgs:
 				# No more images - stop filling out the grid.
 				break
 			img = imgs[img_idx]
 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 			yoff = (img_shape[0] + border) * i
 			xoff = (img_shape[1] + border) * j
 			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

 	cv2.imwrite(str(loc)+'img_'+'epoch_'+str(cnt)+'_loss_'+str(out_loss) + '.jpg', (tile_img + 1)*127.5)


def img_tile2(loc, cnt, out_loss, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
 	if imgs.ndim != 3 and imgs.ndim != 4:
 		raise ValueError('imgs has wrong number of dimensions.')
 	n_imgs = imgs.shape[0]

 	tile_shape = None
 	# Grid shape
 	img_shape = np.array(imgs.shape[1:3])
 	if tile_shape is None:
 		img_aspect_ratio = img_shape[1] / float(img_shape[0])
 		aspect_ratio *= img_aspect_ratio
 		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
 		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
 		grid_shape = np.array((tile_height, tile_width))
 	else:
 		assert len(tile_shape) == 2
 		grid_shape = np.array(tile_shape)

 	# Tile image shape
 	tile_img_shape = np.array(imgs.shape[1:])
 	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

 	# Assemble tile image
 	tile_img = np.empty(tile_img_shape)
 	tile_img[:] = border_color
 	for i in range(grid_shape[0]):
 		for j in range(grid_shape[1]):
 			img_idx = j + i*grid_shape[1]
 			if img_idx >= n_imgs:
 				# No more images - stop filling out the grid.
 				break
 			img = imgs[img_idx]
 			img = np.array((img + 1) * 127.5, dtype=np.uint8)
 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 			yoff = (img_shape[0] + border) * i
 			xoff = (img_shape[1] + border) * j
 			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

 	cv2.imwrite(str(loc)+'img_'+'epoch_'+str(cnt)+'_loss_'+str(out_loss) + '.jpg', (tile_img))





if __name__ == '__main__':
    train()
    
