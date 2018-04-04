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

def classify(img, correct_class=None, target_class=None, plot=False):
    p = sess.run(probs, feed_dict={image: img})[0]

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]

    if plot: 
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
        plt.show()

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

def brightness_transform(image, min_delta=-0.05, max_delta=0.05):
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

def verify_transformations(img, transform_list=transformations):
    for transform in transform_list:
        transform_image = transform(image)
        transform_example = transform_image.eval(feed_dict={image: img})
        classify(transform_example, correct_class=img_class, plot=True)

def eot_adversarial_synthesizer(img, epsilon=8.0/255.0, lr=2e-1, steps=300, target=924):
    """
    synthesis a robust adversarial example with EOT (expectation over transformation) algorithm, Athalye et al. 

    :param epsilon: allowed error 
    :param lr: learning rate 
    :param target: target imagenet class, default is 924 'guacamole'
    """
    x = tf.placeholder(tf.float32, (299, 299, 3))

    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())

    labels = tf.one_hot(y_hat, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
    optim_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, var_list=[x_hat])

    epsilon = tf.placeholder(tf.float32, ())

    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    # initialization step
    sess.run(assign_op, feed_dict={x: img})

    # projected gradient descent
    for i in range(demo_steps):
        # gradient descent step
        _, loss_value = sess.run(
            [optim_step, loss],
            feed_dict={learning_rate: demo_lr, y_hat: demo_target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i+1) % 10 == 0:
            print('step %d, loss=%g' % (i+1, loss_value))

    adv = x_hat.eval() # retrieve the adversarial example

    classify(adv, correct_class=img_class, target_class=demo_target)

    num_samples = 10
    average_loss = 0
    for i in range(num_samples):
        rotated_logits, _ = inception(rotated, reuse=True)
        average_loss += tf.nn.softmax_cross_entropy_with_logits(
            logits=rotated_logits, labels=labels) / num_samples
    optim_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(average_loss, var_list=[x_hat])

    # initialization step
    sess.run(assign_op, feed_dict={x: img})

    # projected gradient descent
    for i in range(demo_steps):
        # gradient descent step
        _, loss_value = sess.run(
            [optim_step, average_loss],
            feed_dict={learning_rate: demo_lr, y_hat: demo_target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i+1) % 50 == 0:
            print('step %d, loss=%g' % (i+1, loss_value))
        
    adv_robust = x_hat.eval() # retrieve the adversarial example
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate robust adversarial example which survives real world perturbations')
    # parser.add_argument('image', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    parser.add_argument('--verify', action='store_true',
                        help='run each individual transformation to verify output')
    parser.add_argument('--image', type=str, 
                        help='input image for adversarial synthesis, one will be chosen at random if not provided')
    args = parser.parse_args()

    # Load script location for loading dependencies
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

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
    
    # HARD CODED EXPECTED IMAGE CLASS
    img_class = 281

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

        if args.verify: 
            verify_transformations(img) 
