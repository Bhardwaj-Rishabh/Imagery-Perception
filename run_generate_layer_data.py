import numpy as np
import tensorflow as tf
import pickle
import vgg16
import utils

img = []

for i in range(1,9):
    img.append(utils.load_image("./Images/png2jpg/F"+str(i)+".jpg"))

for i in range(1,9):
    img.append(utils.load_image("./Images/png2jpg/H"+str(i)+".jpg"))

batch = []
               
for i in img:
    batch.append(i.reshape((1, 224, 224, 3)))

batch = np.concatenate(batch, 0)
               
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [16, 224, 224, 3])
        feed_dict = {images: batch}
        vgg = vgg16.Vgg16()
        
        with tf.name_scope("content_vgg"):
            vgg.build(images)
        
        arr_pool3 = sess.run(vgg.pool3, feed_dict = feed_dict)
        s3 = arr_pool3.shape
        arr_pool3 = np.reshape(arr_pool3,(s3[0], s3[1]*s3[2]*s3[3]))
        
        arr_pool4 = sess.run(vgg.pool4, feed_dict = feed_dict)
        s4 = arr_pool4.shape
        arr_pool4 = np.reshape(arr_pool4,(s4[0], s4[1]*s4[2]*s4[3]))
        
        arr_pool5 = sess.run(vgg.pool5, feed_dict = feed_dict)
        s5 = arr_pool5.shape
        arr_pool5 = np.reshape(arr_pool5,(s5[0], s5[1]*s5[2]*s5[3]))
        
        arr_f6 = sess.run(vgg.fc6,feed_dict=feed_dict)
        arr_f7 = sess.run(vgg.fc7,feed_dict=feed_dict)
        arr_f8 = sess.run(vgg.fc8,feed_dict=feed_dict)


pickle.dump(arr_pool3, open("./layer_data/pool3.p","wb"))
pickle.dump(arr_pool4, open("./layer_data/pool4.p","wb"))
pickle.dump(arr_pool5, open("./layer_data/pool5.p","wb"))
pickle.dump(arr_f6, open("./layer_data/fc_6.p","wb"))
pickle.dump(arr_f7, open("./layer_data/fc_7.p","wb"))
pickle.dump(arr_f8, open("./layer_data/fc_8.p","wb"))
