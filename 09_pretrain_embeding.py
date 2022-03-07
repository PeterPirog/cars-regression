# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
# https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/#converting-text-into-numbers
# https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html

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

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from helper_functions import setSeed  # , loadCarData

# Equivalent to the two lines above
from tensorflow.keras import mixed_precision
from tools.general_tools import nll_loss, embedings2tsv_files, create_training_plots
from tools.domain_tools import loadCarData

import math

policy = mixed_precision.Policy('bfloat16')  #
# mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


def convert_nll_2_mu_std(y_pred, invert_log10=True):
    mu = tf.slice(y_pred, [0, 0], [-1, 1])
    sigma = tf.math.exp(tf.slice(y_pred, [0, 1], [-1, 1]))

    if invert_log10:
        val_max = tf.pow(10.0, mu + sigma)
        mu = tf.pow(10.0, mu)
        sigma = val_max - mu
    return mu, sigma


def nll_mape_metric(y_true, y_pred,k=2.0):
   mu, sigma=convert_nll_2_mu_std(y_pred, invert_log10=True)
   y_true=tf.pow(10.0,y_true)
   mape=100*k*sigma/y_true
   metric = tf.reduce_mean(mape, axis=0)
   return  metric


if __name__ == '__main__':
    # ray.init()

    # SETTINGS
    batch_size = 32
    source_file = 'text_dataset.csv'
    filepath_vect_model = "tokenizer_model_5k"  # source to dir with tokenizer model
    # These values must be the same in tokenizer
    max_vocab_length = 5000
    # Maximum number of tokens in sentence
    max_length = 29
    # Path to save complete regression model
    regression_model_path = 'regression_model'

    setSeed(seed=42, threads=64)
    """
    train_ds, val_ds, train_sentences, val_sentences, \
    train_labels, val_labels = loadCarData(source_file=source_file, number_rows=None,
                                           return_form='full', batch_size=batch_size)
    """
    # for each in val_ds.take(1):
    #    print(each)

    # Load tokenizer model

    loaded_model = tf.keras.models.load_model(filepath_vect_model)
    loaded_vectorizer = loaded_model.layers[0]

    # Define embeding layer
    embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                                 output_dim=128,  # set size of embedding vector
                                 embeddings_initializer="uniform",  # default, intialize randomly
                                 input_length=max_length,  # how long is each input
                                 name="embedding_1")

    filepath_embeding_model='/home/ppirog/projects/cars-regression/regression_model'
    embeding_model=tf.keras.models.load_model(filepath_embeding_model,
                                              custom_objects = {"_tf_keras_metric": nll_loss})

    print(embeding_model)

    """

    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = loaded_model(inputs)
    embedings = embedding(x)
    x = tf.keras.layers.Lambda(lambda y: tf.cast(y, tf.dtypes.bfloat16))(x)
    x = layers.GlobalAveragePooling1D()(embedings)
    outputs = layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_model")
    model_embedings = tf.keras.Model(inputs=inputs, outputs=embedings, name="embedings_model")

    model.summary()

    # Compile model
    model.compile(loss=nll_loss,  # "mse"
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[nll_mape_metric])

   

    # Create callbacks
    # python -m tensorboard.main --logdir logs --bind_all --port=12301
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 profile_batch='20, 40')

    my_callbacks = [tb_callback,
                    tf.keras.callbacks.EarlyStopping(patience=3),
                    tf.keras.callbacks.ModelCheckpoint(filepath=regression_model_path,
                                                       monitor="val_loss",
                                                       save_best_only=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6)
                    ]

    # Fit the model
    history = model.fit(train_ds,  # train_sentences,
                        # train_labels,
                        epochs=5,
                        shuffle=False,
                        batch_size=batch_size,
                        validation_data=val_ds,  # (val_sentences, val_labels),
                        callbacks=my_callbacks)

    model_embedings.save('model_embedings', save_format="tf")
    # model.save("regression_model")
    result = model.evaluate(val_sentences, val_labels)

    print(history.history)

    create_training_plots(history, metric_name='nll_mape_metric', loss_name='loss')

    # Get the weight matrix of embedding layer
    # (these are the numerical patterns between the text in the training dataset the model has learned)
    embed_weights = model.get_layer("embedding_1").get_weights()[0]
    print(embed_weights.shape)  # same size as vocab size and embedding_dim (each word is a embedding_dim size vector)

    words_in_vocab = loaded_vectorizer.get_vocabulary()

    # Save embedings to tsv file
    embedings2tsv_files(embed_weights_from_layer=embed_weights,
                        words_in_vocab=words_in_vocab,
                        embedding_vectors_file_name="embedding_vectors_tmp.tsv",
                        embedding_metadata_file_name="embedding_metadata_tmp.tsv")

    predictions = model.predict(train_sentences)
    mu, sigma = convert_nll_2_mu_std(predictions)
    print(mu, sigma)
    """