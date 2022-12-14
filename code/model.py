import tensorflow as tf
from keras import regularizers
import hyperparameters as hp

class Style_Detector_Model(tf.keras.Model):
    def __init__(self):
        super(Style_Detector_Model, self).__init__()
        num_classes = 5

        self.optimizer = tf.keras.optimizers.Adam(hp.learning_rate)

        self.architecture = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3,strides=(2,2), padding="same", name="conv_layer1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, activation="relu", padding="same", name="block1_conv1"),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, activation="relu", padding="same", name="block1_conv2"),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), padding="same"),
			tf.keras.layers.Dropout(rate=0.2),
			
			tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, activation="relu", padding="same", name="block2_conv1"),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, activation="relu", padding="same", name="block2_conv2"),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), padding="same"),
			tf.keras.layers.Dropout(rate=0.2),

			tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation="relu", padding="same", name="block3_conv1"),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation="relu", padding="same", name="block3_conv2"),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), padding="same"),
			tf.keras.layers.Dropout(rate=0.2),
			
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(128, activation="relu"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(128, activation="relu"),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
    
    def call(self, x):
        return self.architecture(x)