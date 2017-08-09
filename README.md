# Semantic Segmentation Project

This is the elective project involving semantic segmentation for Udacity's 
self-driving car program.  The original project repo can be found 
[here](https://github.com/udacity/CarND-Semantic-Segmentation).

![Example image](/images/example.png)



### Training Data

The default training data is:

 * [Kitti](http://www.cvlibs.net/datasets/kitti/eval_road.php)


Another options is:

 * [Cityscapes](https://www.cityscapes-dataset.com/)


These should be extracted to the `data` folder.  In the case of the `Cityscapes`
data, the project may need to be manipulated in order to be compatible.



### Running the Project

The project can be run with ease by invoking `python main.py`.  The way it 
is set up in this repo, using a GTX 1060 it takes about 15 minutes to 
train.



### Initialization and Tests

When run, the project invokes some tests around building block functions that 
help ensure tensors of the right size and description are being used.  I 
updated test method test_safe() to report the name of the function invoked 
since this is a bit more useful with the tests.  I also cleaned up the test 
code and allowed for printing rather than suppressing it, as this aids in ad
hoc debugging.  The project has been restructured in a class to facilitate 
better passing of hyperparameters for training; more on this in a later
section.

Once the tests are complete, the code checks for the existance 
of a base VGG16 model, and if not found downloads it.  Once it is available,
it is used as the basis for being reconnected with skip layers and recapping
the model with an alternative top end for semantic segmentation.  



### Defining Skip Layers in VGG16

Essentially what is done is layers 3, 4 and 7 are connected as skip layers.  The
reason for this is we want to have more size independence of features, and using
skip-layers helps attain this.  This aspect of the code can be seen in the 
`layers()` function.  The existing layers to the point of the classifier are 
frozen and the top layer is replaced with a new pixel-wise classifier with two 
output classes per pixel.  Reason for this is we want to train a semantic segmentation
mask.  We do this with cross-entropy loss on flattened data for the logits 
and labels.  This can be seen in the `optimize()` function.



### Training 

At this point the network is trained.  This is pretty normal, though I added `tqdm` 
so that there is a little nicer output reporting while going through batches and 
epochs.  This can be seen in the `train_nn()` function.  The only thing that is a 
little curious about it is getting the numeric values for `keep_prob` and 
`learning_rate` into the feed dictionary.  The reason for this is these are expected
to be floating point numbers, are arguments to the function, but are passed as tensors.  This
is a little curious to me since it makes setting configurations a little bit less than
ideal in terms of code cleanliness.  To overcode this, we need to either add *more* 
arguments to the already excessive functions, or significantly change how the 
tests are written to accommodate.  I don't like it, but I put them as configurable
values in the feed dictionary.

Once the training is complete, there is a handy helper function for running some
images through the inference step.  Not much to say about this, the code is a one-liner 
that depends on the `save_inference_sample()` function.  There are some images below
of what the results are.

The model is also saved.  This is accomplished by using a TensorFlow `Saver()`.  The
metadata and the graph definition are written out.  Reason for this is because we
can use an optimizer on the graph for use in faster inference applications.  This wasn't
directly germane to the goal of the exercise, but it is useful to know how to do.



### Results

The results are surprisingly good.  The image at the top is some of the output.  Of
the hundreds of test images, there are very few that do not have fairly adequate 
road coverage.  Places where it seems to fail most include:

 * Small regions, such as between cars or around bicycles
 * Wide expanses of road with poorly defined boundaries
 * Road forks with dominant lane lines

Surprisingly, it works very well at distinguishing between roads and intersecting
railroad tracks.  This is probably related to why it does not segment the road at
intersections with dominant lane lines as well, as the high contrast parallel lines
have few examples in the training set.



### Code Style

TensorFlow code tends to be a bit annoying since it can be difficult to 
compose a bunch of functions that generate the computation graph elegantly.  This 
is generally further complicated by teasing out layers to reconnect for transfer
learning.  

While the notion of having tests is excellent, it further complicates the 
cleanliness of the code, as it ends up with functions of surprising numbers
of arguments, many of which are somewhat token arguments like placeholders.  Though
the benefit is, like all testing, demonstration that something is what is expected 
and there is no regression, in this case it makes for some very messy code and
some dependency issues.  For example, to maintain the existing tests without adding
to the arguments for the `train_nn()` function, the `keep_prob` and `learning_rate`
are embedded in the function.  I pursued making a `FCN` class that encapsulates 
the functionality implemented, but since every method has `self` as an argument
this drammatically changes the way the tests are implemented.


