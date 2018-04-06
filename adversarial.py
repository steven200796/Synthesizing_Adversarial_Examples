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
import matplotlib.pyplot as plt

# Image Loading
import PIL

# Optional Flags
import argparse

# 3.14159265
import numpy as np
import random
import scipy.misc

from datetime import datetime
savedir = None

def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

def load_image(img_path):
    img = PIL.Image.open(img_path)
    big_dim = max(img.width, img.height)
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
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
    
    label_score_pairs = "\n".join(['{:^20s} : {:05.3f}{!s}'.format(imagenet_labels[i], p[i], " target" if i == target_class else " correct " if i == correct_class else "") for i in topk])
    if target_class:
        print('Adversarial Results {:s}'.format(tag))
    else:
        print('Initial Results {:s}'.format(tag))
    print(label_score_pairs + '\n')
    return label_score_pairs

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
def brightness_transform(image, min_delta=-0.3, max_delta=0.3):
    brightened_img = tf.clip_by_value(tf.image.adjust_brightness(image, tf.random_uniform((), minval=min_delta, maxval=max_delta)), 0, 1)
    return brightened_img

def gaussian_noise_transform(image, min_noise=0, max_noise=0.1):
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=max_noise, dtype=tf.float32) 
    noisy_img = tf.clip_by_value(noise + image, 0, 1)
    return noisy_img

def translation_transform(image, min_translate=-80, max_translate=80):
   translated_img = tf.contrib.image.translate(image, tf.stack([tf.random_uniform((), minval=min_translate, maxval=max_translate), tf.random_uniform((), minval=min_translate, maxval=max_translate)]))
   return translated_img

transformations = [rotate_transform, scale_transform, brightness_transform, gaussian_noise_transform, translation_transform]

def verify_transformations(img, correct_class, target_class, plot=True, save=False, transform_list=transformations):
    for transform in transform_list:
        transform_image = transform(image)
        transform_example = transform_image.eval(feed_dict={image: img})
        classify(transform_example, correct_class=correct_class, target_class=target_class, plot=plot, save=save, tag=transform.__name__)
    # scramble the transforms and apply them all
    for i in range(5):
        transform_example = np.copy(img)
        for transform in sorted(transform_list, key=lambda x: random.random()):
            transform_image = transform(transform_example)
            transform_example = transform_image.eval(feed_dict={image: transform_example})
        classify(transform_example, correct_class=correct_class, target_class=target_class, plot=plot, save=save, tag='Composition_{:d}'.format(i))
        
# Sampling function for training
'''
Uniformly random choice of single transformations does not extend to compositions

scrambled compositions of all transformations simply did not converge
'''
def sample_transformations(image, n=1):
    transform_image = image
    scrambled_transformations = sorted(transformations, key=lambda x: random.random())
    if n != 0:
        scrambled_transformations = scrambled_transformations[0:n]
    for transform in scrambled_transformations:
        transform_image = transform(transform_image)
    return transform_image

#eps=8.0/255.0, lr=2e-1, steps=300, target=924
def eot_adversarial_synthesizer(img, eps=35.0/255.0, lr=2e-1, steps=300, target=924):
    """
    synthesis a robust adversarial example with EOT (expectation over transformation) algorithm, Athalye et al. 

    :param eps: allowed error 
    :param lr: learning rate 
    :param target: target imagenet class, default is 924 'guacamole'
    """

    """ Tensorflow Portion """
    x = tf.placeholder(tf.float32, (299, 299, 3))

    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])

    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    num_samples = 10
    average_loss = 0
    for i in range(num_samples):
        transformed = sample_transformations(image)
        transformed_logits, _ = inception(transformed, reuse=True)
        average_loss += tf.nn.softmax_cross_entropy_with_logits(
            logits=transformed_logits, labels=labels) / num_samples

    optim_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(average_loss, var_list=[x_hat])

    """ Normal Python Gig """
    # initialization step
    sess.run(assign_op, feed_dict={x: img})

    #for j in range(len(transformations)):
        # projected gradient descent
    for i in range(steps):
        # gradient descent step
        _, loss_value, _ = sess.run(
            [optim_step, average_loss, transformed],
            feed_dict={learning_rate: lr, y_hat: target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: eps})
        if (i+1) % 50 == 0:
            print('step %d, loss=%g' % (i+1, loss_value))
        
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

    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        # Load Pretrained InceptionV3 Model
        data_dir = tempfile.mkdtemp()
        inception_tarball, _ = urlretrieve(
            'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
        tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

        # Load InceptionV3 
        image = tf.Variable(tf.zeros((299, 299, 3))) 
        logits, probs = inception(image, reuse=False)
        
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

        #TODO add target_class support
        adversarial_img = eot_adversarial_synthesizer(img)
        label_score_pairs = classify(adversarial_img, correct_class=args.correct_class, target_class=args.target_class, plot=not args.noplot, save=args.save, tag='base')
        # Verify adversariality maintains under transformations
        verify_transformations(adversarial_img, args.correct_class, args.target_class, plot=not args.noplot, save=args.save)
        if args.save:
            scipy.misc.imsave(os.path.join(savedir, 'adversary.jpeg'), adversarial_img)
            with open(os.path.join(savedir,'adversarial_classification_scores.txt'), 'w') as f:
                f.write(label_score_pairs)

