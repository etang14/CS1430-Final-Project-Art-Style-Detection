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
            tf.keras.layers.Conv2D(filters=64, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same", kernel_regularizer=regularizers.l2(l=0.01)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
            kernel_regularizer=regularizers.l2(l=0.01)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
            kernel_regularizer=regularizers.l2(l=0.01)),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
            kernel_regularizer=regularizers.l2(l=0.01)),
            tf.keras.layers.Conv2D(filters=1028, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
            kernel_regularizer=regularizers.l2(l=0.01)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation='leaky_relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='leaky_relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, x):
        return self.architecture(x)