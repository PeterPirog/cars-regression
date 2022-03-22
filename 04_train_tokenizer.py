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

