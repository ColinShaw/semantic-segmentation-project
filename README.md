# Semantic Segmentation Project

This is the elective project involving semantic segmentation for Udacity's 
self-driving car program.  The original project repo can be found 
[here](https://github.com/udacity/CarND-Semantic-Segmentation).



### Training Data

The default training data is:

 * [Kitti](http://www.cvlibs.net/datasets/kitti/eval_road.php)


Another options is:

 * [Cityscapes](https://www.cityscapes-dataset.com/)



### Running the Project

The project can be run with ease by invoking `python main.py`.

When run, the project invokes some tests that help ensure tensors of the 
right size and description are being used.  I updated test method test_safe()
to report the name of the function invoked since this is a bit more useful
with the tests.  Once the tests are complete, the code checks for the existance 
of a base VGG16 model, and if not found downloads it.  Once it is available,
it is used as the basis for the project.  

Essentially what is done is layers 3, 4 and 7 are connected as skip layers.  The
reason for this is we want to have more size independence of features, and using
skip-layers helps attain this.  This aspect of the code can be seen in the 
`layers()` function.  The existing layers to the point of the classifier are 
frozen and the top layer is replaced with a new pixel-wise classifier with two 
output classes per pixel.  Reason for this is we want to train a semantic segmentation
mask.  We do this with cross-entropy loss on flattened data for the logits 
and labels.  This can be seen in the `optimize()` function.

At this point the network is trained.  This is pretty normal, though I added `tqdm` 
so that there is a little nicer output reporting while going through batches and 
epochs.  This can be seen in the `train_nn()` function.  The only thing that is a 
little curious about it is getting the numeric values for `keep_prob` and 
`learning_rate` into the feed dictionary.  The reason for this is these are expected
to be floating point numbers, are arguments to the function, but are defined in the
tests as tensors.  Tensors generate the error "the value of a feed cannot be a tf.Tensor object,"
which leads one to just put the constant value in the function.  It isn't immediately
apparent what the goal is in having a function like `train_nn()` that accepts arguments
that would seem to be floating point values but have the tests written with 
placeholders.  Placeholders are inconvenient to get the value from, and the whole
point of the feed dictionary is to inject values into placeholders.  I just changed that
in the tests, as it seemed more sensible in my implementation.  I also removed the 
print suppression function, which makes it hard to do ad hoc manual debugging by 
printing.

Once the training is complete, there is a handy helper function for running some
images through the inference step.  Not much to say about this, the code is a one-liner 
that depends on the `save_inference_sample()` function.  There are some images below
of what the results are.

The model is also saved.  This is accomplished by using a TensorFlow `Saver()`.  The
metadata and the graph definition are written out.  Reason for this is because we
can use an optimizer on the graph for use in faster inference applications.  This wasn't
directly germane to the goal of the exercise, but it is useful to know how to do.



### Results



### Code Review

TensorFlow code tends to be a bit annoying since it can be annoying to elegantly
compose a bunch of functions that generate the computation graph.  Abstractions
like are found here are nice because they facilitate testing, but they are somewhat
annoying because they end up creating a bunch of functions taking a lot of arguments 
simply because placeholders are needed for some of the TensorFlow operations.  This
really gets messy.  I like it that tests were written to help with being on the 
right track while developing the code, but some of the functions are just messy.

