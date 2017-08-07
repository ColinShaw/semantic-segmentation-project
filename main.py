import os.path
import tensorflow as tf
import helper
import warnings
import project_tests as tests
from tqdm import tqdm


def load_vgg(sess, vgg_path):
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph       = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob   = graph.get_tensor_by_name('keep_prob:0')
    l3          = graph.get_tensor_by_name('layer3_out:0')
    l4          = graph.get_tensor_by_name('layer4_out:0')
    l7          = graph.get_tensor_by_name('layer7_out:0')
    return input_image, keep_prob, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    l3_trans = tf.layers.conv2d_transpose(vgg_layer3_out, 4096, 5, 16, 'same')
    l4_trans = tf.layers.conv2d_transpose(vgg_layer4_out, 4096, 5, 8, 'same')
    l7_out   = tf.add(tf.add(l3_trans, l4_trans), vgg_layer7_out)
    out      = tf.layers.conv2d_transpose(l7_out, num_classes, 1, 1)
    return out
tests.test_layers(layers)


def optimize_cross_entropy(nn_last_layer, correct_label, learning_rate, num_classes):
    logits               = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label        = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss   = tf.reduce_mean(cross_entropy_logits)
    train_op             = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize_cross_entropy)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    for epoch in tqdm(range(epochs)):
        for image, label in get_batches_fn(batch_size):
            print(image)
            print(learning_rate)
            print(keep_prob)
            feed_dict = { 
                'input_image'   : image,
                'correct_label' : label,
                'keep_prob'     : keep_prob,
                'learning_rate' : learning_rate
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            print(loss)
tests.test_train_nn(train_nn)


def run():
    epochs         = 1
    batch_size     = 5
    learning_rate  = 0.01
    num_classes    = 2
    image_shape    = (160, 576)
    data_dir       = './data'
    runs_dir       = './runs'
    vgg_path       = os.path.join(data_dir, 'vgg')
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    helper.check_compatibility() 
    tests.test_for_kitti_dataset(data_dir)
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:

        # Initialize globals and placeholders
        sess.run(tf.global_variables_initializer())
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))

        # Define network 
        input_image, keep_prob, l3, l4, l7   = load_vgg(sess, vgg_path)
        output                               = layers(l3, l4, l7, num_classes)
        logits, train_op, cross_entropy_loss = optimize_cross_entropy(output, correct_label, learning_rate, num_classes)
       
        # Train the model 
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save images using the helper
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # Save the model
        s = tf.train.Saver()
        s.save(sess, 'model/final.ckpt')
        s.export_meta_graph('model/final.meta')
        tf.train.write_graph(sess.graph_def, "./model/", "final.pb", False)


if __name__ == '__main__':
    run()
