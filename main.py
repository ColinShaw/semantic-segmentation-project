import math
import os.path
import tensorflow as tf
import helper
import project_tests as tests
from tqdm import tqdm


'''
Load the VGG16 model
'''
def load_vgg(sess, vgg_path):
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

tests.test_load_vgg(load_vgg, tf)


''' 
Define the layers
'''
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    l7    = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    l7_up = tf.layers.conv2d_transpose(l7, num_classes, 4, 2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    l4    = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    l4_l7 = tf.add(l7_up, l4)
    l4_up = tf.layers.conv2d_transpose(l4_l7, num_classes, 4, 2, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    l3    = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    l3_l4 = tf.add(l4_up, l3)
    l3_up = tf.layers.conv2d_transpose(l3_l4, num_classes, 16, 8, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01))
    return l3_up

tests.test_layers(layers)


'''
Optimizer based on cross entropy
'''
def optimize_cross_entropy(nn_last_layer, correct_label, learning_rate, num_classes):
    # Reshape logits and label for computing cross entropy
    logits               = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label        = tf.reshape(correct_label, (-1, num_classes))

    # Compute cross entropy and loss
    cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss   = tf.reduce_mean(cross_entropy_logits)

    # Define a training operation using the Adam optimizer
    train_op             = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize_cross_entropy)


''' 
Training runs
'''
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    # Iterate over epochs
    for epoch in range(epochs):
        print("Epoch: " + str(epoch) + "/" + str(epochs))

        # Iterate over the batches using the batch generation function
        batch      = get_batches_fn(batch_size)
        size       = math.ceil(289 / batch_size)
        total_loss = []

        for i, d in tqdm(enumerate(batch), desc="Batch", total=size):

            # Create the feed dictionary
            image, label = d
            feed_dict = { 
                input_image   : image,
                correct_label : label,
                keep_prob     : 0.5,
                learning_rate : 0.00005
            }

            # Train and compute the loss
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            total_loss.append(loss)

        # Compute mean epoch loss
        mean_loss = sum(total_loss) / size
        print("Loss:  " + str(loss) + "\n")
tests.test_train_nn(train_nn)


'''
Save the model
'''
def save_model(sess):
    saver = tf.train.Saver()
    saver.save(sess, 'data/model.ckpt')
    saver.export_meta_graph('data/model.meta')
    tf.train.write_graph(sess.graph_def, "./data/", "model.pb", False)


'''
Main routine
'''
def run():
    # Define some project constants
    epochs         = 20
    batch_size     = 10
    num_classes    = 2
    image_shape    = (160, 576)
    data_dir       = './data'
    runs_dir       = './runs'
    vgg_path       = os.path.join(data_dir, 'vgg')

    # Generate the batching function
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # Check for compatibility, data set and conditionally download VGG16 model
    helper.check_compatibility() 
    tests.test_for_kitti_dataset(data_dir)
    helper.maybe_download_pretrained_vgg(data_dir)

    # TensorFlow session
    with tf.Session() as sess:

        # Placeholders
        learning_rate = tf.placeholder(dtype = tf.float32)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))

        # Define network and training operations 
        input_image, keep_prob, l3, l4, l7   = load_vgg(sess, vgg_path)
        output                               = layers(l3, l4, l7, num_classes)
        logits, train_op, cross_entropy_loss = optimize_cross_entropy(output, correct_label, learning_rate, num_classes)
      
        # Initialize variables 
        sess.run(tf.global_variables_initializer())

        # Train the model 
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save images using the helper
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # Save the model
        save_model(sess)


'''
Entry point
'''
if __name__ == '__main__':
    run()

