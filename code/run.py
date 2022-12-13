import numpy as np
import hyperparameters as hp
import json
import pickle
import os
from os.path import isfile, join
import tensorflow as tf
import datetime
from matplotlib import pyplot as plt

def LIME_explainer(model, path, preprocess_fn):
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
    image = preprocess_fn(image)
    

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

def main():

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
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

    class_names = train_ds.class_names
    print(class_names)

    
    num_classes = 5
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.Conv2D(16, 3, activation='leaky_relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='leaky_relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='leaky_relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation='leaky_relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='leaky_relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60
    )


if __name__ == '__main__':
    main()