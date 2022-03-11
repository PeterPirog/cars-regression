# https://docs.ray.io/en/master/train/examples/tensorflow_mnist_example.html

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



if __name__ == '__main__':
    batch_size=32
    embeded_train_sentences= np.load('/home/ppirog/projects/cars-regression/npdata/embeded_train_sentences.npy')
    embeded_val_sentences = np.load('/home/ppirog/projects/cars-regression/npdata/embeded_val_sentences.npy')

    train_labels = np.load('/home/ppirog/projects/cars-regression/npdata/train_labels.npy')
    val_labels = np.load('/home/ppirog/projects/cars-regression/npdata/val_labels.npy')


    print(embeded_val_sentences,np.shape(embeded_val_sentences))



    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(128,), dtype=tf.float32)

    outputs = layers.Dense(1, activation="linear")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model1")

    model.summary()

    # Compile model
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['mse'])

    # Fit the model
    history = model.fit(x=embeded_train_sentences,y=train_labels,
                        batch_size=batch_size,
                        epochs=5,
                        shuffle=True,
                        validation_data=(embeded_val_sentences,val_labels),
                        callbacks=None,
                        workers=16,
                        use_multiprocessing=True)


