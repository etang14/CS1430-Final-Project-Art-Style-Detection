from datetime import datetime
import tensorflow as tf

from model import Style_Detector_Model
import hyperparameters as hp

def main():

    time_now = datetime.now()
    timestamp = time_now.strftime("%d-%m-%y-%H%M%S")
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

    model = Style_Detector_Model()

    checkpoint_path = "checkpoints/" + timestamp
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

    callback_list = [
        cp_callback,
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/",
            update_freq="batch",
            profile_batch=0
        )
    ]

    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callback_list
    )

if __name__ == '__main__':
    main()