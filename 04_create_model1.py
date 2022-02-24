from sklearn.model_selection import train_test_split
#import pandas as pd
import modin.pandas as pd
import joblib
import json
import ray


import tensorflow as tf
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(64)

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# from pyearth import Earth

if __name__ == '__main__':
    ray.init()
    filepath = '/home/ppirog/projects/cars-regression/text_dataset.csv'
    sep = ';'
    encoding = 'utf-8'
    print('Read csv')
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    dataset=tf.data.experimental.make_csv_dataset(
        file_pattern=filepath, batch_size=32, column_names=None, column_defaults=None,
        label_name='Amount', select_columns=None, field_delim=';',
        use_quote_delim=True, na_value='', header=True, num_epochs=None,
        shuffle=True, shuffle_buffer_size=10000, shuffle_seed=42,
        prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
        num_rows_for_inference=100, compression_type=None, ignore_errors=False
    )

    # Line to reduce dataset size for tests
    #df = df.iloc[:5000]

    # Use train_test_split to split training data into training and validation sets
    print('split dataframe to sets')
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                # dedicate 10% of samples to validation set
                                                                                random_state=42)  # random state for reproducibility




    # Note: in TensorFlow 2.6+, you no longer need "layers.experimental.preprocessing"
    # you can use: "tf.keras.layers.TextVectorization", see https://github.com/tensorflow/tensorflow/releases/tag/v2.6.0 for more

    max_vocab_length=1000
    max_length=30


    # Use the default TextVectorization variables
    text_vectorizer = TextVectorization(max_tokens=None,
                                        # how many words in the vocabulary (all of the different words in your text)
                                        standardize="lower_and_strip_punctuation",  # how to process text
                                        split="whitespace",  # how to split tokens
                                        ngrams=None,  # create groups of n-words?
                                        output_mode="int",  # how to map tokens to numbers
                                        output_sequence_length=None)  # how long should the output sequence of tokens be?
    # pad_to_max_tokens=True) # Not valid if using max_tokens=None

    embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                                 output_dim=128,  # set size of embedding vector
                                 embeddings_initializer="uniform",  # default, intialize randomly
                                 input_length=max_length,  # how long is each input
                                 name="embedding_1")

    # Build model with the Functional API

    inputs = layers.Input(shape=(1,), dtype="string")  # inputs are 1-dimensional strings
    x = text_vectorizer(inputs)  # turn the input text into numbers
    x = embedding(x)  # create an embedding of the numerized numbers
    x = layers.GlobalAveragePooling1D()(x)  # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
    outputs = layers.Dense(1, activation="linear")(x)  # create the output layer, want binary outputs so use sigmoid activation
    model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")  # construct the model

    # Compile model
    model_1.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    # Get a summary of the model
    model_1.summary()

    # Fit the model
    model_1_history = model_1.fit(train_sentences,
                                  train_labels,
                                  epochs=5,
                                  validation_data=(val_sentences, val_labels),
                                  callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                                         experiment_name="simple_dense_model")])


    # Total number of words from train_sentences

    # Fit the text vectorizer to the training text
    print('Adapt vectorizer')
    text_vectorizer.adapt(train_sentences)

    # Get the unique words in the vocabulary
    words_in_vocab = text_vectorizer.get_vocabulary()
    top_5_words = words_in_vocab[:5]  # most common tokens (notice the [UNK] token for "unknown" words)
    bottom_5_words = words_in_vocab[-5:]  # least common tokens
    print(f"Words in vocab: {words_in_vocab}")
    print(f"Number of words in vocab: {len(words_in_vocab)}")
    print(f"Top 5 most common words: {top_5_words}")
    print(f"Bottom 5 least common words: {bottom_5_words}")

    """
    """