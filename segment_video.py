import numpy as np
import scipy.misc
import tensorflow as tf
from tqdm import tqdm 
from moviepy.editor import *


class SegmentVideo(object):

    '''
    Constants
    '''
    image_shape = (160, 576)


    '''
    Constructor with param setting
    '''
    def __init__(self, params):
        for p in params:
            setattr(self, p, params[p])


    '''
    Segments the image
    '''
    def segment_image(self, image):
        image = scipy.misc.imresize(image, self.image_shape)
        feed_dict = {
            self.keep_prob:   1.0,
            self.input_image: [image]
        }
        run_op       = tf.nn.softmax(self.logits)
        im_softmax   = self.sess.run([run_op], feed_dict=feed_dict)
        im_softmax   = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask         = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask         = scipy.misc.toimage(mask, mode="RGBA")
        street_im    = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        return np.array(street_im)


    '''
    Main processing loop
    '''
    def process_video(self):
        new_frames = []
        video      = VideoFileClip(self.input_video)
        for frame in video.iter_frames():
            new_frame = self.segment_image(frame)
            new_frames.append(new_frame)
        new_video = ImageSequenceClip(new_frames, fps=video.fps)
        new_video.write_videofile(self.output_video, audio=False)


    '''
    Restore model and retrieve pertinent tensors
    '''
    def restore_model(self):
        saver = tf.train.import_meta_graph('data/model.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('data/'))
        graph = tf.get_default_graph()
        self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        self.input_image = graph.get_tensor_by_name('image_input:0')
        self.logits      = graph.get_tensor_by_name('logits:0')


    '''
    Run the segmentation
    '''
    def run(self):
        self.sess = tf.Session() 
        self.restore_model()
        self.process_video()



'''
Entry point
'''
if __name__=='__main__':
    params = {
        'input_video':  'video/sunset.mp4',
        'output_video': 'video/segmented.mp4'
    }
    sv = SegmentVideo(params)
    sv.process_video()

