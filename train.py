import math
import os.path
import tensorflow as tf
import helper
import project_tests as tests
from tqdm import tqdm


class FCN(object):

    '''
    Static properties
    '''
    training_images = 289
    num_classes     = 2
    image_shape     = (160, 576)
    data_dir        = './data'
    runs_dir        = './runs'
    training_subdir = 'data_road/training'


    '''
    Constructor for setting params
    '''
    def __init__(self, params):
        for p in params:
            setattr(self, p, params[p])


    '''
    Load the VGG16 model
    '''
    def load_vgg(self, sess, vgg_path):

        # Load the saved model
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

        # Get the relevant layers for constructing the skip-layers out of the graph
        graph       = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        l3          = graph.get_tensor_by_name('layer3_out:0')
        l4          = graph.get_tensor_by_name('layer4_out:0')
        l7          = graph.get_tensor_by_name('layer7_out:0')
        return input_image, keep_prob, l3, l4, l7

  
    '''
    Truncated norm to make layer initialization readable
    '''
    def tf_norm(self):
        return tf.truncated_normal_initializer(stddev=self.init_sd)


    ''' 
    Define the layers
    '''
    def layers(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

        # 1x1 convolutions of the three layers
        l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=self.tf_norm())
        l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=self.tf_norm())
        l3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=self.tf_norm())
       
        # Upsample layer 7 and add to layer 4
        total = tf.layers.conv2d_transpose(l7,    num_classes,  4, 2, 'SAME', kernel_initializer=self.tf_norm())
        total = tf.add(total, l4)

        # Upsample the sum and add to layer 3
        total = tf.layers.conv2d_transpose(total, num_classes,  4, 2, 'SAME', kernel_initializer=self.tf_norm())
        total = tf.add(total, l3)
  
        # Upsample the total and return
        total = tf.layers.conv2d_transpose(total, num_classes, 16, 8, 'SAME', kernel_initializer=self.tf_norm())
        return total


    '''
    Optimizer based on cross entropy
    '''
    def optimize_cross_entropy(self, nn_last_layer, correct_label, learning_rate, num_classes):

        # Reshape logits and label for computing cross entropy
        logits        = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
        correct_label = tf.reshape(correct_label, (-1, num_classes))

        # Compute cross entropy and loss
        cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
        cross_entropy_loss   = tf.reduce_mean(cross_entropy_logits)

        # Define a training operation using the Adam optimizer
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
        return logits, train_op, cross_entropy_loss


    ''' 
    Define training op
    '''
    def train_nn(self, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):

        # Iterate over epochs
        for epoch in range(epochs):
            print("Epoch: " + str(epoch) + "/" + str(epochs))

            # Iterate over the batches using the batch generation function
            total_loss = []
            batch      = get_batches_fn(batch_size)
            size       = math.ceil(self.training_images / batch_size)

            for i, d in tqdm(enumerate(batch), desc="Batch", total=size):

                # Create the feed dictionary
                image, label = d
                feed_dict = { 
                    input_image   : image,
                    correct_label : label,
                    keep_prob     : self.dropout,
                    learning_rate : self.learning_rate
                }

                # Train and compute the loss
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
                total_loss.append(loss)

            # Compute mean epoch loss
            mean_loss = sum(total_loss) / size
            print("Loss:  " + str(loss) + "\n")


    '''
    Save the model
    '''
    def save_model(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, 'data/model.ckpt')
        saver.export_meta_graph('data/model.meta')
        tf.train.write_graph(sess.graph_def, "./data/", "model.pb", False)


    '''
    Run the tests
    '''
    def run_tests(self):
        tests.test_load_vgg(self.load_vgg, tf)
        tests.test_layers(self.layers)
        tests.test_optimize(self.optimize_cross_entropy)
        tests.test_train_nn(self.train_nn)

 
    '''
    Main training routine
    '''
    def run(self):

        # Check for compatibility, data set and conditionally download VGG16 model
        helper.check_compatibility() 
        tests.test_for_kitti_dataset(self.data_dir)
        helper.maybe_download_pretrained_vgg(self.data_dir)

        # Define static project constants
        vgg_path      = os.path.join(self.data_dir, 'vgg')
        training_path = os.path.join(self.data_dir, self.training_subdir)

        # Define the batching function
        get_batches_fn = helper.gen_batch_function(training_path, self.image_shape)

        # TensorFlow session
        with tf.Session() as sess:

            # Placeholders
            learning_rate = tf.placeholder(dtype = tf.float32)
            correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, self.num_classes))

            # Define network and training operations 
            input_image, keep_prob, l3, l4, l7   = self.load_vgg(sess, vgg_path)
            output                               = self.layers(l3, l4, l7, self.num_classes)
            logits, train_op, cross_entropy_loss = self.optimize_cross_entropy(output, correct_label, learning_rate, self.num_classes)
      
            # Initialize variables 
            sess.run(tf.global_variables_initializer())

            # Train the model 
            self.train_nn(sess, self.epochs, self.batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

            # Save images using the helper
            helper.save_inference_samples(self.runs_dir, self.data_dir, sess, self.image_shape, logits, keep_prob, input_image)

            # Save the model
            self.save_model(sess)


'''
Entry point
'''
if __name__ == '__main__':
    params = {
        'learning_rate':   0.0001,
        'dropout':         0.5,
        'epochs':          5,
        'batch_size':      1,
        'init_sd':         0.01
    }
    fcn = FCN(params)
    fcn.run_tests()
    fcn.run()

