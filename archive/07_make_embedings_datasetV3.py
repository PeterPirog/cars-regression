# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
# https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/#converting-text-into-numbers
# https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html
# https://www.tensorflow.org/guide/keras/train_and_evaluate?hl=en#training_evaluation_from_tfdata_datasets

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# import pandas as pd
import modin.pandas as pd
import joblib
import json
import ray
import numpy as np
import os
from datetime import datetime

import tensorflow as tf
import random
import modin.pandas as pdm

tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(64)

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from helper_functions import setSeed, loadCarData, load_trained_vectorizer,get_dataset_partitions_tf

if __name__ == '__main__':
    ray.init()

    setSeed(seed=42, threads=64)
    # file_pattern = '/home/ppirog/projects/cars-regression/text_dataset.csv'
    # df = pdm.read_csv(file_pattern, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

    # df = df.rename(columns={'Text': 'input_1'}, inplace=False)
    # df.to_csv(path_or_buf='/home/ppirog/projects/cars-regression/text_dataset2.csv',index=False)

    batch_size = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    file_pattern = '/home/ppirog/projects/cars-regression/text_dataset2.csv'

    # Get size of dataset (number of rows)
    df = pdm.read_csv(file_pattern, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)
    ds_size=df.shape[0]



    ds = tf.data.experimental.make_csv_dataset(
        file_pattern, batch_size,
        label_name='Amount', column_defaults=[tf.string, tf.float32],
        field_delim=',', ignore_errors=True, num_epochs=1,shuffle_seed=42,
        prefetch_buffer_size=AUTOTUNE, num_parallel_reads=AUTOTUNE)

    train_size=10000
    val_size=2000

    train_ds = ds.take(train_size).batch(batch_size,drop_remainder=True)   #.batch(batch_size).cache().prefetch(AUTOTUNE)
    val_ds = ds.skip(train_size).take(val_size).batch(batch_size,drop_remainder=True)

    #dataset = dataset.cache()

    #dataset = dataset.prefetch(AUTOTUNE)
    #ds = ds.batch(batch_size,drop_remainder=True, num_parallel_calls=AUTOTUNE)


    #train_ds, val_ds, test_ds=get_dataset_partitions_tf(ds, ds_size,
    #                                                    train_split=0.0028,
    #                                                    val_split=0.00028,
    #                                                    test_split=0.99692,
    #                                                    shuffle=True,
    #                                                    seed=42,
    #                                                    #batch_size=batch_size,
    #                                                    shuffle_size=10000)

    #train_ds=train_ds.batch(batch_size)
    #val_ds=val_ds.batch(batch_size)
    #test_ds=test_ds.batch(batch_size)





    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern, batch_size, column_names=['Text','Amount'], column_defaults=['string','float32'],
        label_name='Amount', select_columns=None, field_delim=';',
        use_quote_delim=True, na_value='', header=True, num_epochs=None,
        shuffle=True, shuffle_buffer_size=10000, shuffle_seed=42,
        prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
        num_rows_for_inference=100, compression_type=None, ignore_errors=True)
    """

    #for each in dataset.take(1):
    #    print(each)

    #print(ds.element_spec)

    # print(f'train_sentences shape: {train_sentences.shape}')

    # Load tokenizer model
    dirpath_vect_model = "tokenizer_model_10k"
    loaded_model,loaded_vectorizer=load_trained_vectorizer(dirpath_vect_model)
    #loaded_model = tf.keras.models.load_model(filepath_vect_model)
    #loaded_vectorizer = loaded_model.layers[0]

    # These values must be the same in tokenizer
    max_vocab_length = 10000
    max_length = 29
    output_dim = 64

    # Define embeding layer
    embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                                 output_dim=output_dim,  # set size of embedding vector
                                 embeddings_initializer="uniform",  # default, intialize randomly
                                 input_length=max_length,  # how long is each input
                                 name="embedding_1")

    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = loaded_model(inputs)
    embedings = embedding(x)
    x = layers.GlobalAveragePooling1D()(embedings)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_model")
    model_embedings = tf.keras.Model(inputs=inputs, outputs=embedings, name="embedings_model")

    model.summary()

    # Compile model
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mse"])

    # Create callbacks
    # python -m tensorboard.main --logdir logs --bind_all --port=12301
    monitor = "loss"
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 profile_batch='20, 40')

    my_callbacks = [tb_callback,
                    tf.keras.callbacks.EarlyStopping(patience=10, monitor=monitor),
                    tf.keras.callbacks.ModelCheckpoint(filepath='regression_model',
                                                       monitor=monitor,  # "val_loss"
                                                       save_best_only=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,  # "val_loss"
                                                         factor=0.1, patience=6, min_lr=1e-6)
                    ]

    # Fit the model
    history = model.fit(train_ds,
                        epochs=5,
                        batch_size=batch_size,
                        validation_data=val_ds,
                        callbacks=my_callbacks
                        )

    # model_embedings.save('model_embedings', save_format="tf")
    # model.save("regression_model")
    # result = model.evaluate(val_sentences, val_labels)

    # result = model.predict(train_sentences)

    # print(result)
    # print(np.shape(result))
    # Create mse and loss plots

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

"""

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









    train_sentences, val_sentences, train_labels, val_labels = loadCarData(number_rows=10000)
    # train X & y
train_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_features.values, tf.string)
) 
train_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_targets.values, tf.int64),

) 
# test X & y
test_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_features.values, tf.string)
) 
test_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_targets.values, tf.int64),

train_ds = tf.data.Dataset.zip(
    (
            train_text_ds,
            train_cat_ds_raw
     )
) 

test_ds = tf.data.Dataset.zip(
    (
            test_text_ds,
            test_cat_ds_raw
     )
) 

batch_size = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
buffer_size= train_ds.cardinality().numpy()

train_ds = train_ds.shuffle(buffer_size=buffer_size)\
                   .batch(batch_size=batch_size,drop_remainder=True)\
                   .cache()\
                   .prefetch(AUTOTUNE)

test_ds = test_ds.shuffle(buffer_size=buffer_size)\
                   .batch(batch_size=batch_size,drop_remainder=True)\
                   .cache()\
                   .prefetch(AUTOTUNE)


print(train_ds.element_spec)

    """
