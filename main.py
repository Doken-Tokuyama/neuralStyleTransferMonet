#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:20:14 2019

@author: admin
"""

import os
from matplotlib.pyplot import imshow
import nst_utils as util
import tensorflow as tf
import scipy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tf.reset_default_graph()
model = util.load_vgg_model("imagenet-vgg-verydeep-19.mat")

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

content_image = scipy.misc.imread("louvre.jpg")
content_image = scipy.misc.imresize(content_image, size=(225,300,3))
content_image = util.reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("monet.jpg")
style_image = scipy.misc.imresize(style_image, size=(225,300,3))
style_image = util.reshape_and_normalize_image(style_image)

generated_image = util.generate_noise_image(content_image)
imshow(generated_image[0])
imshow(content_image[0])
imshow(style_image[0])
    
# Assign the content image to be the input of the VGG model.  
# Start interactive session
sess = tf.InteractiveSession()
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = util.compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = util.compute_style_cost(model, STYLE_LAYERS, sess)
J = util.total_cost(J_content, J_style, alpha = 10, beta = 40)

# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 30):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%1 == 0:
            Jc = sess.run(J_content)
            Js = sess.run(J_style)
            Jt = sess.run(J)
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            util.save_image("out/" + str(i) + ".png", generated_image)
    
    # save last generated image
    util.save_image('out/generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)