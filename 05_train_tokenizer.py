# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow

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
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

    #df=df.iloc[:5000]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                random_state=42)

    print(f'train_sentences shape: {train_sentences.shape}')

    max_vocab_length=2000 #21660
    max_length=29


    # Use the default TextVectorization variables
    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                        # how many words in the vocabulary (all of the different words in your text)
                                        standardize="lower_and_strip_punctuation",  # how to process text
                                        split="whitespace",  # how to split tokens
                                        ngrams=None,  # create groups of n-words?
                                        output_mode="int",  # how to map tokens to numbers
                                        output_sequence_length=max_length,
                                        name='tokenizer_layer')  # how long should the output sequence of tokens be?

    text_vectorizer.adapt(train_sentences)
    vocab = text_vectorizer.get_vocabulary()  # To get words back from token indices
    print(f"Dictionary has: {len(vocab)} words")

    #Save vocabulary to file
    with open('vocabulary_2k.json', 'w') as fp:
        json.dump(vocab, fp)


    # Create model.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(text_vectorizer)

    # Save.
    filepath = "tokenizer_model_2k"
    model.save(filepath, save_format="tf")

    # Load.
    loaded_model = tf.keras.models.load_model(filepath)
    loaded_vectorizer = loaded_model.layers[0]

    test_sentence="bodykombimpv communebiałystok communebiałobrzegi communebe³¿yce communebe³chatów communebelskduży communebarlinek"
    test_sentence=train_sentences[:2]
    print(loaded_vectorizer(test_sentence))
    print(text_vectorizer(test_sentence))

    """
    df_train_sentences=pd.DataFrame(train_sentences)

    df.to_csv(path_or_buf='df_train_sentences.csv', sep=';', encoding='utf-8', index=False)


    

    dataset=tf.data.experimental.make_csv_dataset(
        file_pattern='/home/ppirog/projects/cars-regression/text_dataset_small.csv', batch_size=3, column_names=None, column_defaults=None,
        label_name='Amount', select_columns=None, field_delim=';',
        use_quote_delim=True, na_value='', header=True, num_epochs=None,
        shuffle=True, shuffle_buffer_size=10000, shuffle_seed=42,
        prefetch_buffer_size=None, num_parallel_reads=tf.data.AUTOTUNE, sloppy=False,
        num_rows_for_inference=100, compression_type=None, ignore_errors=False
    )

    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    #for item in dataset.take(5):
    #    print(item[0])



    
    
    
    
    
    
    
    text_vectorizer.adapt(train_sentences)

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
                    metrics=["mse"])

    # Get a summary of the model
    model_1.summary()

    # Fit the model
    model_1_history = model_1.fit(train_sentences, val_sentences,
                                  epochs=5,
                                  validation_data=(train_labels, val_labels))



    sep = ';'
    encoding = 'utf-8'
    print('Read csv')
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
    
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
    
    
        sep = ';'
    encoding = 'utf-8'
    print('Read csv')
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
    df=df.iloc[:10000]
    df.to_csv(path_or_buf='/home/ppirog/projects/cars-regression/text_dataset_small.csv', sep=sep, encoding=encoding, index=False)

    """
