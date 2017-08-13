import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import cv2
from glob import glob
from urllib.request import urlretrieve
from distutils.version import LooseVersion
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def check_compatibility():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))
    if not tf.test.gpu_device_name():
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def maybe_download_pretrained_vgg(data_dir):
    vgg_filename = 'vgg.zip'
    vgg_path     = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')
    ]
    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    def get_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images    = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image         = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image      = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # Augment rotation
                angle         = np.random.uniform(-30, 30)
                image         = scipy.misc.imrotate(image, angle)
                gt_image      = scipy.misc.imrotate(gt_image, angle)

                # Augment luminance
                image         = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
                image[:,:,1]  = image[:,:,1] * (0.25 + np.random.uniform(0.25, 0.75))
                image         = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

                # Augment translation
                x             = np.random.uniform(-40, 40)
                y             = np.random.uniform(-40, 40)
                image         = cv2.warpAffine(image, np.float32([[1,0,x],[0,y,0]]), image_shape)
                gt_image      = cv2.warpAffine(gt_image, np.float32([[1,0,x],[0,y,0]]), image_shape)
          
                gt_bg         = np.all(gt_image == background_color, axis=2)
                gt_bg         = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image      = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image)
            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image        = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        im_softmax   = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
        im_softmax   = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask         = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask         = scipy.misc.toimage(mask, mode="RGBA")
        street_im    = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

