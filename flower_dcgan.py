
# coding: utf-8

# In[1]:


#Imports
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from PIL import Image


# In[2]:


#Import paths
dir_path = './flower_images'
image_name_list = os.listdir(dir_path)
image_path_list = [os.path.join(dir_path, img) for img in image_name_list]


# In[3]:


#Load image func
def getImage(path):
    img = Image.open(path)
    img.thumbnail((96, 96))
    img.convert('RGB')
    img = np.asarray(img, dtype = np.float32)/255
    img = img[:,:,:3]
    return img
        


# In[4]:


#Get the images as ndarrays
images = np.array([getImage(img) for img in image_path_list])


# In[5]:


plt.imshow(images[0])


# In[6]:


#The placeholders for training
def Inputs(width, height, chann, z_dim):
    
    x = tf.placeholder(tf.float32, shape=(None, width, height, chann))
    z = tf.placeholder(tf.float32, shape=(None, z_dim))
    learning_rate = tf.placeholder(tf.float32)
    
    return x, z, learning_rate


# In[7]:


#The discriminator, he will judge how bad the generator image is and using leakyrelu for activation

def Discriminator(image, reuse):
    
    with tf.variable_scope('Discriminator', reuse = reuse):
        alpha = 0.2
        x = tf.layers.conv2d(image, 16, 5,                              kernel_initializer = tf.random_normal_initializer(stddev = 0.01),                             bias_initializer = tf.random_normal_initializer(stddev = 0.01), strides = 2, padding = 'same')
        bn = tf.layers.batch_normalization(x, training = True)
        lrelu = tf.maximum(x*alpha, x) #LeakyRelu
        #48x48x16
        layer1 = tf.layers.conv2d(lrelu, 32, 5,                  kernel_initializer = tf.random_normal_initializer(stddev = 0.01),                                   bias_initializer = tf.random_normal_initializer(stddev = 0.01), strides = 2, padding = 'same')
        bn1 = tf.layers.batch_normalization(layer1, training = True)
        lrelu1 = tf.maximum(bn1 * alpha, bn1)
        #24x24x32
        layer2 = tf.layers.conv2d(lrelu1, 96, 5,                                   kernel_initializer = tf.random_normal_initializer(stddev = 0.01),                                   bias_initializer = tf.random_normal_initializer(stddev = 0.01), strides = 2,                                  padding = 'same')
        bn2 = tf.layers.batch_normalization(layer2, training = True)
        lrelu2 = tf.maximum(bn2 * alpha, bn2)
        #12x12x96
        flatten = tf.reshape(lrelu2, [-1, 12*12*96])
        logit = tf.layers.dense(flatten, 1)
        output = tf.nn.sigmoid(logit)
        return output, logit
        


# In[8]:


#The Generator will generate the samples using conv2d_transpose
def Generator(image_z, is_train):
    with tf.variable_scope('Generator', reuse = not is_train):
        alpha = 0.2
        x = tf.layers.dense(image_z, 12*12*96)
        x = tf.reshape(x, [-1, 12, 12, 96])
        x = tf.nn.relu(x)
        
        layer1 = tf.layers.conv2d_transpose(x, 32, 5, strides = 2, padding = 'same', activation = tf.nn.relu)
        #24x24x32
        layer2 = tf.layers.conv2d_transpose(layer1, 16, 5, strides = 2, padding = 'same', activation = tf.nn.relu)
        #48x48x16
        layer3 = tf.layers.conv2d_transpose(layer2, 3, 5, strides = 2, padding = 'same', activation = tf.nn.relu)
        #96x96x3
        logit = tf.nn.tanh(layer3)
        
        return logit


# In[9]:


def Loss(image_real, image_z):
    
    gen_images = Generator(image_z, True) #Sample Image
    dis_real, dis_logit_real = Discriminator(image_real, False) #Input the real Images to the discriminator
    dis_fake, dis_logit_fake = Discriminator(gen_images, True) #Input the samples to the discriminator
    
    #The Loss for the the fake images
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_logit_fake,                                                                           labels = tf.zeros_like(dis_fake)))
    #The loss for the real images
    dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_logit_real,\ 
                                                                          labels = tf.ones_like(dis_real)*0.9))
    #The generator loss
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_logit_fake,                                                                     labels = tf.ones_like(dis_fake)))
    #The discriminator loss
    d_loss = dis_loss_fake + dis_loss_real
    
    return d_loss, gen_loss


# In[10]:


def Optimizer(d_loss, g_loss, learning_rate):
    
    t_vars = tf.trainable_variables()
    #Get the generator weights
    g_vars = [var for var in t_vars if var.name.startswith('Generator')]
    #Get the discriminator loss
    d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
    
    #Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.5).minimize(d_loss, var_list = d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.5).minimize(g_loss, var_list = g_vars)
    
    return d_opt, g_opt


# In[11]:


#Hyperparameters
n_epochs = 5001
learning_rate = 0.0001
batch_size = 32
z_dim = 500
width = 96
height = 96
chan = 3


# In[ ]:


tf.reset_default_graph()

#Instantiate the functions
image_real, image_z, lnr = Inputs(width, height, chan, z_dim)
d_loss, g_loss = Loss(image_real, image_z)
d_opt, g_opt = Optimizer(d_loss, g_loss, lnr)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    #Initialize everything
    init.run()
    n_batches = int(len(images)/batch_size)
    batches = np.array_split(images, n_batches)
    print("Starting....")
    
    #Train the network
    for epoch in range(n_epochs):
        
        for ii in range(n_batches):
            
            batch = batches[ii]
            batch_z = np.random.uniform(-1., 1, size=(batch.shape[0], z_dim))
            
            feed = {image_real:batch, image_z:batch_z, lnr:learning_rate}
            
            dis_train_loss = d_loss.eval({image_real:batch, image_z:batch_z})
            gen_train_loss = g_loss.eval({image_real:batch, image_z:batch_z})
            
            _ = sess.run(d_opt, feed_dict = feed)
            _ = sess.run(g_opt, feed_dict = feed)
        
        if epoch % 50 == 0:
            print("Epoch:{}    Discriminator Loss:{}     Generator Loss:{}".format(epoch, dis_train_loss, gen_train_loss))

            f, axis = plt.subplots(2, 8, figsize = (13, 3))
            batch_examples = np.random.uniform(-1., 1., size=(16, z_dim))
            gen_images = sess.run(Generator(image_z, False), feed_dict = {image_z: batch_examples})
            for i in range(8):
                axis[0][i].imshow(gen_images[i])
                axis[0,i].axis('off')
                axis[1][i].imshow(gen_images[i+8])
                axis[1,i].axis('off')
            plt.show()
        #Save the sess
        if epoch % 100 == 0:
            saver = tf.train.Saver()
            saving = saver.save(sess, "".join(('./logs/checkPoint',str(epoch),'.ckpt')))
        
        
            
        

