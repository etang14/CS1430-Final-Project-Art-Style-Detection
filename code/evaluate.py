from PIL import Image
from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from skimage.transform import resize
from model import Style_Detector_Model

import tensorflow as tf
import hyperparameters as hp
import numpy as np

def LIME_explainer(model, path):
    """
    This function takes in a trained model and a path to an image and outputs 5
    visual explanations using the LIME model
    """

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)
        plt.show()

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")
    plt.show()


def visualize_layer(model, img_path, layer_name, figsize, nrows=4, ncols=4, view_img=True):
    image = Image.open(img_path)
    image = image.resize((224,224))
    image = np.array(image)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    curr_layer = model.get_layer("sequential").get_layer(layer_name).output
    slice_model  = tf.keras.Model(inputs=model.get_layer("sequential").input, outputs=curr_layer)
    slice_output = slice_model.predict(image[None,:,:,:])

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            curr_ax = axes[row, col]
            out = slice_output[0,:,:,idx].astype(np.uint8)    
            out = Image.fromarray(out)
            out = out.resize(image.shape[:-1], resample=Image.BOX)
            curr_ax.imshow(out)
            if view_img:
                curr_ax.imshow(image, alpha=0.3)

    return fig, axes

model = Style_Detector_Model()
model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

data_dir = "data/images"
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels = 'inferred',
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(hp.img_size, hp.img_size),
        batch_size=hp.batch_size
    )

loss, acc = model.evaluate(val_ds, verbose=2)
print("New model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights("checkpoints/14-12-22-145812")

loss, acc = model.evaluate(val_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

image_paths = [
    "data/images/abstract-expressionism/231511.jpg",
    "data/images/cubism/223799.jpg",
    "data/images/impressionism/246332.jpg",
    "data/images/renaissance/186388.jpg",
    "data/images/romanticism/364902.jpg",
]
for path in image_paths:
    image = imread(path)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    print(model.predict(image[None,:,:,:]))

    visualize_layer(
        model=model, img_path=path, layer_name="conv_layer1", figsize=(15,15)
    )

    LIME_explainer(
        model=model, path=path
    )