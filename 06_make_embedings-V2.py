# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
# https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/#converting-text-into-numbers

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# import pandas as pd
import modin.pandas as pd
import joblib
import json
import ray
import numpy as np

import tensorflow as tf

tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(64)

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

if __name__ == '__main__':
    ray.init()
    filepath = '/home/ppirog/projects/cars-regression/text_dataset.csv'
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

    df = df.iloc[:5000]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                random_state=42)

    print(f'train_sentences shape: {train_sentences.shape}')

    # Load tokenizer model
    filepath_vect_model = "tokenizer_model_10k"
    loaded_model = tf.keras.models.load_model(filepath_vect_model)
    loaded_vectorizer = loaded_model.layers[0]

    # These values must be the same in tokenizer
    max_vocab_length = 10000
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
    x = layers.GlobalAveragePooling1D()(
        embedings)  # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="relu")(
        x)  # create the output layer, want binary outputs so use sigmoid activation
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_model")
    model_embedings = tf.keras.Model(inputs=inputs, outputs=embedings, name="embedings_model")

    model.summary()

    # Compile model
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mse"])

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath='regression_model',
                                           monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6)
    ]

    # Fit the model
    history = model.fit(train_sentences,
                        train_labels,
                        epochs=500,
                        validation_data=(val_sentences, val_labels),
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
