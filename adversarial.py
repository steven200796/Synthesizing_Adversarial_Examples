#!/usr/bin/env python3

# Load InceptionV3 Model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

# Load Pretrained Weights Into InceptionV3 
import tempfile
from urllib.request import urlretrieve
import tarfile
import os

# Load Imagenet Classes
import json

# Load Plotting Module
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Image Loading
import PIL 
import skimage.color as color

# Optional Flags
import argparse

# 3.14159265
import numpy as np
import random
import scipy.misc

from datetime import datetime

''' Tensorflow Bug https://github.com/tensorflow/tensorflow/issues/12071 '''
from tensorflow.python.framework import function
@function.Defun(tf.float32, tf.float32)
def norm_grad(x, dy):
    return dy*(x/tf.norm(x))

@function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
    return tf.norm(x)


savedir = None

def rgb2labnorm(img):
    img = color.rgb2lab(img)
    #normalize
    img = (img + [0, 128, 128]) / [100, 255, 255]
    return img

def labnorm2rgb(img):
    #denormalize
    img = (img *  [100, 255, 255]) - [0, 128, 128]
    img = color.lab2rgb(img)
    return img

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return tf.reshape(srgb_pixels, tf.shape(lab))

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

def load_image(img_path, lab=False):
    img = PIL.Image.open(img_path)
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299)) 
    if lab:
        return rgb2labnorm(img)
    else:
        img = (np.asarray(img) / 255.0).astype(np.float32)
        return img

def classify(img, correct_class=None, target_class=None, plot=False, save=False, tag=''):
    p = sess.run(probs, feed_dict={image: img})[0]

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]

    if plot or save: 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.sca(ax1)
        ax1.imshow(img)
        fig.sca(ax1)
        
        barlist = ax2.bar(range(10), topprobs)
        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk:
            barlist[topk.index(correct_class)].set_color('g')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10),
                   [imagenet_labels[i][:15] for i in topk],
                   rotation='vertical')
        fig.subplots_adjust(bottom=0.2)
        if save:
            if target_class:
                plt.savefig(os.path.join(savedir, "adversary_results{!s}.jpeg".format('_'+tag if tag else tag)))
            else:
                plt.savefig(os.path.join(savedir, "initial_results.jpeg"))
        if plot:
            plt.show()
    
    label_score_pairs = ['{:^20s} : {:05.3f}{!s}'.format(imagenet_labels[i], p[i], " target" if i == target_class else " correct " if i == correct_class else "") for i in topk]
    caption = '{:s} {:s}'.format('Adversarial Results' if target_class else 'Initial Results', tag)
    label_score_pairs.insert(0, caption)
    print_string = '\n'.join(label_score_pairs)
    print(print_string + '\n')
    return print_string

def rotate_transform(image, min_angle=-np.pi, max_angle=np.pi):
    rotated_img = tf.contrib.image.rotate(image, tf.random_uniform((), minval=min_angle, maxval=max_angle))
    return rotated_img

def scale_transform(image, min_scale=0.9, max_scale=1.4):
    scale = tf.random_uniform((), minval=min_scale, maxval=max_scale)
    height = tf.cast(tf.multiply(scale, 299), tf.int32)
    width = tf.cast(tf.multiply(scale, 299), tf.int32)
    scaled_img = tf.image.resize_images(image, tf.stack([height, width])) 

    scaled_img = tf.image.resize_image_with_crop_or_pad(scaled_img, 299, 299)
    return scaled_img

# -0.05 and 0.05
def brightness_transform(image, min_delta=-0.05, max_delta=0.05):
    brightened_img = tf.clip_by_value(tf.image.adjust_brightness(image, tf.random_uniform((), minval=min_delta, maxval=max_delta)), 0, 1)
    return brightened_img

def gaussian_noise_transform(image, min_noise=0, max_noise=0.1):
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=max_noise, dtype=tf.float32) 
    noisy_img = tf.clip_by_value(noise + image, 0, 1)
    return noisy_img

def translation_transform(image, min_translate=-80, max_translate=80, interpolation='NEAREST'):
    tx = tf.random_uniform((), minval=min_translate, maxval=max_translate) 
    ty = tf.random_uniform((), minval=min_translate, maxval=max_translate)
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tf.contrib.image.transform(image, transforms, interpolation)

#def translation_transform(image, min_translate=-80, max_translate=80):
#   translated_img = tf.contrib.image.translate(image, tf.stack([tf.random_uniform((), minval=min_translate, maxval=max_translate), tf.random_uniform((), minval=min_translate, maxval=max_translate)]))
#   return translated_img

transformations = [scale_transform, rotate_transform, brightness_transform, gaussian_noise_transform, translation_transform]

def verify_transformations(img, correct_class, target_class, plot=True, save=False, transform_list=transformations):
    results = []
    for transform in transform_list:
        transform_image = transform(img)
        transform_example = transform_image.eval()
        results.append(classify(transform_example, correct_class=correct_class, target_class=target_class, plot=plot, save=save, tag=transform.__name__))
    # scramble the transforms and apply them all
    for i in range(5):
        transform_example = np.copy(img)
        for transform in transformations:
            #sorted(transform_list, key=lambda x: random.random()):
            transform_image = transform(transform_example)
            transform_example = transform_image.eval()
        results.append(classify(transform_example, correct_class=correct_class, target_class=target_class, plot=plot, save=save, tag='Composition_{:d}'.format(i)))
    return "\n\n".join(results)
        
# Sampling function for training
'''
Uniformly random choice of single transformations does not extend to compositions

scrambled compositions of all transformations simply did not converge

Pretty sure it's the scrambling which won't converge so will try with regular compositions
'''
def sample_transformations(image, n=1):
    transform_image = image
    #scrambled_transformations = sorted(transformations, key=lambda x: random.random())
    scrambled_transformations = transformations
    if n != 0:
        scrambled_transformations = scrambled_transformations[0:n]
    for transform in scrambled_transformations:
        transform_image = transform(transform_image)
    return transform_image

# eps=8.0/255.0, lr=2e-1, steps=300, target=924
def eot_adversarial_synthesizer(img, eps=8/255.0, lr=3e-4, steps=500, target=924, restore=False, saver=None, save=False):
    """
    synthesis a robust adversarial example with EOT (expectation over transformation) algorithm, Athalye et al. 

    :param eps: allowed error 
    :param lr: learning rate 
    :param target: target imagenet class, default is 924 'guacamole'
    """

    """ Tensorflow Portion """
    x = tf.placeholder(tf.float32, (299, 299, 3))  
    #trainable adversarial input
    # hacky way of linking adversarial model to inceptionv3 model
    x_hat = image
    assign_op = tf.assign(x_hat, x)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)

    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    num_samples = 10 
    average_loss = 0

    for i in range(num_samples):
        for transform in transformations:
            transformed = transform(image)
            transformed_logits, _ = inception(transformed, reuse=True)
            average_loss += tf.nn.softmax_cross_entropy_with_logits(
                logits=transformed_logits, labels=labels) / (num_samples * len(transformations))

    temp = set(tf.all_variables())
    optim_step = tf.train.AdamOptimizer(learning_rate).minimize(average_loss,var_list=[x_hat]) 
    # Hacky Adam Variables Initializer
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    sess.run(assign_op, feed_dict={x: img})
    if restore:
        # restore adversarial model
        saver.restore(sess, os.path.join(__location__, "tmp/model.ckpt"))

    """ Normal Python Gig """
    # projected gradient descent
    for i in range(steps):
        # gradient descent step
        _, loss_value, = sess.run(
            [optim_step, average_loss],
            feed_dict={learning_rate: lr, y_hat: target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: eps})
        if (i+1) % 50 == 0:
            print('step %d, loss=%g' % (i+1, loss_value))
        if save and (i+1)%100 == 0:
            adv_robust = x_hat.eval() # retrieve the adversarial example 
            scipy.misc.imsave(os.path.join(savedir, 'adversary.jpeg'), adv_robust)

    adv_robust = x_hat.eval() # retrieve the adversarial example
    return adv_robust

def eot_adversarial_synthesizer_lab_lgr(img, eps=8/255.0, lr=2e-1, steps=400, target=924, lagrange_c=0.08, restore=False, saver=None, save=False):
    """
    synthesis a robust adversarial example with EOT (expectation over transformation) algorithm, Athalye et al. 

    :param eps: allowed error 
    :param lr: learning rate 
    :param target: target imagenet class, default is 924 'guacamole'
    :param lagrange_c: lagrangian relaxation constant
    """

    """ Tensorflow Portion """
    x = tf.placeholder(tf.float32, (299, 299, 3))  

    c = tf.placeholder(tf.float32, ())
    # Must denorm x to transform to LAB

    x_lab = tf.Variable(tf.stack(preprocess_lab(rgb_to_lab(x))))

    #trainable adversarial input
    # hacky way of linking adversarial model to inceptionv3 model
    x_hat = image
    assign_op = tf.assign(x_hat, x)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)

  #  epsilon = tf.placeholder(tf.float32, ())
    projected = tf.clip_by_value(x_hat, 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    num_samples = 10 
    average_loss = 0
    average_prob = 0
    average_norm = 0

    for transform in transformations:
        for i in range(num_samples):
            transformed = transform(image)
            _, transformed_prob = inception(transformed, reuse=True)
 
            norm = tf.reduce_sum(tf.square(
                    tf.subtract(
                        x_lab,
                        preprocess_lab(rgb_to_lab(
                        x_hat
                        ))
                        )))

            prob = transformed_prob[0, target]

            average_prob += prob / (num_samples * len(transformations))
            average_norm += norm / (num_samples * len(transformations))
            average_loss += tf.add(
                    tf.negative(tf.log(prob)), 
                    tf.scalar_mul(
                        c, 
                        norm
                        )
                ) / (num_samples * len(transformations))
            

    temp = set(tf.all_variables())
    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(average_loss,var_list=[x_hat]) 

    # Hacky Adam Variables Initializer
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    #initialize variables
    sess.run(x_lab.initializer, feed_dict={x: img})
    sess.run(assign_op, feed_dict={x: img})

    if restore:
        # restore adversarial model
        saver.restore(sess, os.path.join(__location__, "tmp/model.ckpt"))


    """ Normal Python Gig """
    # projected gradient descent
    for i in range(steps):
        # gradient descent step
        _, loss_value, target_prob, d_norm = sess.run(
            [optim_step, average_loss, average_prob, average_norm],
            feed_dict={x: img, c: lagrange_c, learning_rate: lr, y_hat: target})
        # project step
        print('step %d, loss=%g, prob=%g, norm=%g' % (i+1, loss_value, target_prob, d_norm))
        sess.run(project_step)
        if (i+1) % 50 == 0:
            print('step %d, loss=%g, prob=%g, norm=%g' % (i+1, loss_value, target_prob, d_norm))
        if save and (i+1)%100 == 0:
            adv_robust = x_hat.eval() # retrieve the adversarial example 
            scipy.misc.imsave(os.path.join(savedir, 'adversary.jpeg'), adv_robust)

    adv_robust = x_hat.eval() # retrieve the adversarial example
    return adv_robust

def eot_adversarial_synthesizer_lab_lgr_composition(img, eps=8/255.0, lr=8e-1, steps=3000, target=924, lagrange_c=0.03, restore=False, saver=None, save=False):
    """
    synthesis a robust adversarial example with EOT (expectation over transformation) algorithm, Athalye et al. 

    :param eps: allowed error 
    :param lr: learning rate 
    :param target: target imagenet class, default is 924 'guacamole'
    :param lagrange_c: lagrangian relaxation constant
    """

    """ Tensorflow Portion """
    x = tf.placeholder(tf.float32, (299, 299, 3))  

    c = tf.placeholder(tf.float32, ())
    # Must denorm x to transform to LAB

    x_lab = tf.Variable(tf.stack(preprocess_lab(rgb_to_lab(x))))

    #trainable adversarial input
    # hacky way of linking adversarial model to inceptionv3 model
    x_hat = image
    assign_op = tf.assign(x_hat, x)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)

  #  epsilon = tf.placeholder(tf.float32, ())
    projected = tf.clip_by_value(x_hat, 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    num_samples = 30 
    average_loss = 0
    average_log = 0
    average_norm = 0

    for i in range(num_samples):
        transformed = sample_transformations(image, len(transformations))
        _, transformed_prob = inception(transformed, reuse=True)

        norm = tf.reduce_sum(tf.square(
                tf.subtract(
                    x_lab,
                    preprocess_lab(rgb_to_lab(
                    x_hat
                    ))
                    )))

        average_log += tf.nn.softmax_cross_entropy_with_logits(
            logits=transformed_logits, labels=labels) / num_samples 
        average_norm += norm / num_samples
    average_loss += tf.add(
            average_log, 
            tf.scalar_mul(
                c, 
                average_norm
                )
            )  

    temp = set(tf.all_variables())
    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(average_loss,var_list=[x_hat]) 

    # Hacky Adam Variables Initializer
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    #initialize variables
    sess.run(x_lab.initializer, feed_dict={x: img})
    sess.run(assign_op, feed_dict={x: img})

    if restore:
        # restore adversarial model
        saver.restore(sess, os.path.join(__location__, "tmp/model.ckpt"))


    """ Normal Python Gig """
    # projected gradient descent
    for i in range(steps):
        # gradient descent step
        _, loss_value, target_log, d_norm = sess.run(
            [optim_step, average_loss, average_log, average_norm],
            feed_dict={x: img, c: lagrange_c, learning_rate: lr, y_hat: target})
        # project step
        print('step %d, loss=%g, prob=%g, norm=%g' % (i+1, loss_value, target_log, d_norm))
        sess.run(project_step)
        if (i+1) % 50 == 0:
            print('step %d, loss=%g, prob=%g, norm=%g' % (i+1, loss_value, target_log, d_norm))
        if save and (i+1)%100 == 0:
            adv_robust = x_hat.eval() # retrieve the adversarial example 
            scipy.misc.imsave(os.path.join(savedir, 'adversary.jpeg'), adv_robust)

    adv_robust = x_hat.eval() # retrieve the adversarial example
    return adv_robust

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate robust adversarial example which survives real world perturbations')
    # parser.add_argument('image', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    parser.add_argument('--save', action='store_true',
                        help='save adversarial result')
    parser.add_argument('--noplot', action='store_true',
                        help='disable plotting so code execution does not block')
    parser.add_argument('--classify', action='store_true',
                        help='only classify the image with ImageNet, terminate before generation of adversarial example')
    parser.add_argument('--verify', action='store_true',
                        help='run each individual transformation to verify output')
    parser.add_argument('--restore', action='store_true',
                        help='restore last session')
    parser.add_argument('--image', type=str, 
                        help='input image for adversarial synthesis, one will be chosen at random if not provided')
    parser.add_argument('--correct_class', type=int, default=281,
                        help='ground truth class of input image')
    parser.add_argument('--target_class', type=int, default=924,
                        help='targeted adversarial class of input image')

    args = parser.parse_args()

    # Load script location for loading dependencies
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # directory to save results
    if args.save:
        savedir = os.path.join(
        __location__,
        "test_results",
        datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        os.makedirs(savedir)

    # Load Imagenet classes
    imagenet_json = None
    try:
        imagenet_json = os.path.join(__location__, 'imagenet.json')
    except:
        imagenet_json, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/imagenet.json')

    with open(imagenet_json) as f:
        imagenet_labels = json.load(f)

    img_path = None
    if args.image:
        img_path = args.image
    else:
        img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
    
    # img = normal image, image = tensorflow image
    img = load_image(img_path)

    # Load Pretrained InceptionV3 Model
    data_dir = tempfile.mkdtemp()
    inception_tarball, _ = urlretrieve(
        'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

    # Load InceptionV3 
    image = tf.Variable(tf.zeros((299, 299, 3)))
    logits, probs = inception(image, reuse=False)

    # Load Adversarial Model
    #eot_adversarial_synthesizer(image, reuse=False)

    tf.logging.set_verbosity(tf.logging.ERROR)

    # create model 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore InceptionV3 Model
        restore_vars = [
            var for var in tf.global_variables()
            if var.name.startswith('InceptionV3/')
        ]

        saver = tf.train.Saver(restore_vars)
        saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))
        
        # Classify original image, store results
        label_score_pairs = classify(img, correct_class=args.correct_class, plot=not args.noplot, save=args.save)
        if args.save:
            with open(os.path.join(savedir, 'initial_classification_scores.txt'), 'w') as f:
                f.write(label_score_pairs)

        if args.verify: 
            verify_transformations(img, args.correct_class, args.target_class, plot=not args.noplot, save=args.save)

        if args.classify:
            exit()
            
        adversarial_vars=[]
        if args.save or args.restore:
            adversarial_vars = [
                var for var in tf.global_variables()
                if not var.name.startswith('InceptionV3/')
            ]
            saver = tf.train.Saver(adversarial_vars)
        
        #TODO add target_class support
        adversarial_img = eot_adversarial_synthesizer_lab_lgr_composition(img, restore=args.restore, saver=saver, save=args.save)

        #save progress
        save_path = saver.save(sess, os.path.join(__location__, "tmp/model.ckpt"))
        print("adversarial generation checkpoint saved in path: %s" % save_path)
 
        base_results = classify(adversarial_img, correct_class=args.correct_class, target_class=args.target_class, plot=not args.noplot, save=args.save, tag='base')
        # Verify adversariality maintains under transformations, this clobbers the x_hat tensor so make sure to save before
        test_results = verify_transformations(adversarial_img, args.correct_class, args.target_class, plot=not args.noplot, save=args.save)
        if args.save:
            scipy.misc.imsave(os.path.join(savedir, 'adversary.jpeg'), adversarial_img)
            with open(os.path.join(savedir,'adversarial_classification_scores.txt'), 'w') as f:
                f.write(base_results)
                f.write('\n\n')
                f.write(test_results)
