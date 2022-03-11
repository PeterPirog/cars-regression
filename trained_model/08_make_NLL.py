# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
# https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/#converting-text-into-numbers
# https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html

from matplotlib import pyplot as plt
# import pandas as pd
import ray
import numpy as np
from datetime import datetime

import tensorflow as tf

# tf.random.set_seed(42)
# tf.config.threading.set_inter_op_parallelism_threads(64)

from tensorflow.keras import layers

from archive.helper_functions import setSeed, loadCarData

# Equivalent to the two lines above
from tensorflow.keras import mixed_precision

import math


def my_loss(y_true, y_pred):
   # print(f'y_true={y_true}')
   # print(f'y_pred={y_pred}')
    mu = tf.slice(y_pred, [0, 0], [-1, 1])
    sigma = tf.math.exp(tf.slice(y_pred, [0, 1], [-1, 1]))

    a = 1 / (tf.sqrt(2. * math.pi) * sigma)
   # print(f'a={a.numpy()}')
    b1 = tf.math.square(mu - y_true)
   # print(f'b1={b1}')
    b2 = 2 * tf.square(sigma)
   # print(f'b2={b2}')
    b = b1 / b2
    loss = tf.reduce_sum(-tf.math.log(a) + b, axis=0)
    return  loss


policy = mixed_precision.Policy('bfloat16')  #
# mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

if __name__ == '__main__':
    ray.init()
    batch_size = 32

    setSeed(seed=42, threads=64)
    train_sentences, val_sentences, train_labels, val_labels = loadCarData(number_rows=None)  # 100000
    print(f'train_sentences shape: {train_sentences.shape}')
    print(f'train_sentences type: {type(train_sentences)}')

    # train X & y
    train_sentences_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_sentences, tf.string))
    train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.dtypes.bfloat16))

    # test X & y
    val_sentences_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_sentences, tf.string))
    val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.dtypes.bfloat16))

    # combine features qith targets into train and validation datasets
    train_ds = tf.data.Dataset.zip(
        (
            train_sentences_ds,
            train_labels_ds
        ))

    val_ds = tf.data.Dataset.zip(
        (
            val_sentences_ds,
            val_labels_ds
        ))

    # Optimize datasets
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    buffer_size = train_ds.cardinality().numpy()

    train_ds = train_ds.shuffle(buffer_size=buffer_size) \
        .batch(batch_size=batch_size, drop_remainder=True) \
        .cache() \
        .prefetch(AUTOTUNE)

    val_ds = val_ds.shuffle(buffer_size=buffer_size) \
        .batch(batch_size=batch_size, drop_remainder=True) \
        .cache() \
        .prefetch(AUTOTUNE)

    # for each in val_ds.take(1):
    #    print(each)

    # Load tokenizer model
    filepath_vect_model = "tokenizer_model_5k"
    loaded_model = tf.keras.models.load_model(filepath_vect_model)
    loaded_vectorizer = loaded_model.layers[0]

    # These values must be the same in tokenizer
    max_vocab_length = 5000
    max_length = 29

    # Define embeding layer
    embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                                 output_dim=128,  # set size of embedding vector
                                 embeddings_initializer="uniform",  # default, intialize randomly
                                 input_length=max_length,  # how long is each input
                                 name="embedding_1")

    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = loaded_model(inputs)
    embedings = embedding(x)
    x = tf.keras.layers.Lambda(lambda y: tf.cast(y, tf.dtypes.bfloat16))(x)
    x = layers.GlobalAveragePooling1D()(embedings)
  #  x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_model")
    model_embedings = tf.keras.Model(inputs=inputs, outputs=embedings, name="embedings_model")

    model.summary()


    # Compile model
    model.compile(loss=my_loss,  # "mse"
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mse"])

    # Create callbacks
    # python -m tensorboard.main --logdir logs --bind_all --port=12301
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 profile_batch='20, 40')

    my_callbacks = [tb_callback,
                    tf.keras.callbacks.EarlyStopping(patience=3),
                    tf.keras.callbacks.ModelCheckpoint(filepath='regression_model',
                                                       monitor="val_loss",
                                                       save_best_only=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6)
                    ]

    # Fit the model
    history = model.fit(train_ds,  # train_sentences,
                        # train_labels,
                        epochs=500,
                        shuffle=False,
                        batch_size=batch_size,
                        validation_data=val_ds,  # (val_sentences, val_labels),
                        callbacks=my_callbacks)

    model_embedings.save('model_embedings', save_format="tf")
    # model.save("regression_model")
    result = model.evaluate(val_sentences, val_labels)

    # result = model.predict(train_sentences)

    print(result)
    print(np.shape(result))
    # Create mse and loss plots
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Get the weight matrix of embedding layer
    # (these are the numerical patterns between the text in the training dataset the model has learned)
    embed_weights = model.get_layer("embedding_1").get_weights()[0]
    print(embed_weights.shape)  # same size as vocab size and embedding_dim (each word is a embedding_dim size vector)

    # # Code below is adapted from: https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_trained_word_embeddings_and_save_them_to_disk
    import io

    words_in_vocab = loaded_vectorizer.get_vocabulary()
    # # Create output writers
    out_v = io.open("embedding_vectors.tsv", "w", encoding="utf-8")
    out_m = io.open("embedding_metadata.tsv", "w", encoding="utf-8")

    # # Write embedding vectors and words to file
    for num, word in enumerate(words_in_vocab):
        if num == 0:
            continue  # skip padding token
        vec = embed_weights[num]
        out_m.write(word + "\n")  # write words to file
        out_v.write("\t".join([str(x) for x in vec]) + "\n")  # write corresponding word vector to file
    out_v.close()
    out_m.close()
