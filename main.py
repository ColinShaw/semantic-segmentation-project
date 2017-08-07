import os.path
import tensorflow as tf
import helper
import warnings
import project_tests as tests


def load_vgg(sess, vgg_path):
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    g  = tf.get_default_graph()
    i  = g.get_tensor_by_name('image_input:0')
    k  = g.get_tensor_by_name('keep_prob:0')
    l3 = g.get_tensor_by_name('layer3_out:0')
    l4 = g.get_tensor_by_name('layer4_out:0')
    l7 = g.get_tensor_by_name('layer7_out:0')
    return i, k, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    l3_trans = tf.layers.conv2d_transpose(vgg_layer3_out, 4096, 5, 16)
    l4_trans = tf.layers.conv2d_transpose(vgg_layer4_out, 4096, 5, 8)
    l7_out   = tf.add(tf.add(l3_trans, l4_trans), vgg_layer7_out)
    out      = tf.layers.conv2d_transpose(l7_out, num_classes, 1, 1)
    return out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits           = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label    = tf.reshape(correct_label, (-1, num_classes))
    cross_ent_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_ent_loss   = tf.reduce_mean(cross_ent_logits)
    _, iou_op        = tf.metrics.mean_iou(correct_label, logits, num_classes)
    return logits, iou_op, cross_ent_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
tests.test_train_nn(train_nn)


def run():
    helper.check_compatibility() 
    learning_rate = 0.01
    num_classes   = 2
    image_shape   = (160, 576)
    data_dir      = './data'
    runs_dir      = './runs'
    tests.test_for_kitti_dataset(data_dir)
    helper.maybe_download_pretrained_vgg(data_dir)
    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: Build NN using load_vgg, layers, and optimize function
        i, k, l3, l4, l7 = load_vgg(sess, vgg_path)
        o                = layers(l3, l4, l7, num_classes)
        #logits, op, loss = optimize(l7, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
