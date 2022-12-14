"""
Number of epochs. If you experiment with more complex networks you
might need to increase this. Likewise if you add regularization that
slows training.
"""
num_epochs = 50

"""
A critical parameter that can dramatically affect whether training
succeeds or fails. The value for this depends significantly on which
optimizer is used. Refer to the default learning rate parameter
"""
learning_rate = 1e-4

"""
Resize image size for task 1. Task 3 must have an image size of 224,
so that is hard-coded elsewhere.
"""
img_size = 224

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 400


"""
The number of image scene classes. Don't change this.
"""
num_classes = 15

batch_size = 50
