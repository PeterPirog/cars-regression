# https://colab.research.google.com/drive/1_hiUXcX6DwGEsPP2iE7i-HAs-5HqQrSe?usp=sharing
# https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
# https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/#converting-text-into-numbers

from sklearn.model_selection import train_test_split
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

    #df=df.iloc[:5000]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                random_state=42)

    print(f'train_sentences shape: {train_sentences.shape}')

    max_vocab_length = 1000  # 21660
    max_length = 29

    # Use the default TextVectorization variables
    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                        # how many words in the vocabulary (all of the different words in your text)
                                        standardize="lower_and_strip_punctuation",  # how to process text
                                        split="whitespace",  # how to split tokens
                                        ngrams=None,  # create groups of n-words?
                                        output_mode="int",  # how to map tokens to numbers
                                        output_sequence_length=max_length)  # how long should the output sequence of tokens be?

    text_vectorizer.adapt(train_sentences)
    vocab = text_vectorizer.get_vocabulary()  # To get words back from token indices
    print(f"Dictionary has: {len(vocab)} words")
    print(f"Dictionary is: {vocab}")

    # Save vocabulary to file
    # with open('vocabulary_5k.json', 'w') as fp:
    #    json.dump(vocab, fp)

    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    output = text_vectorizer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="text_model")

    model.summary()


    test_sentense=train_sentences[0]
    print(f'test sentece: {test_sentense}')
    #result=model.predict(['C_slug_dj Package_p3 Body_kombi Brand_MAZDA Fuel_type_d Gearbox_type_m Generation_type_CX-7_09-15 Import_status_b LPG_system_a Model_type_CX-7 Subject_type_a'])
    result=model.predict(train_sentences)

    print(result)
    print(np.shape(result))
    """
    # Save.
    filepath_vect_model = "tokenizer_model"
    # model.save(filepath, save_format="tf")

    # Load.
    loaded_model = tf.keras.models.load_model(filepath_vect_model)
    loaded_vectorizer = loaded_model.layers[0]

    test_sentence = "bodykombimpv communebiałystok communebiałobrzegi communebe³¿yce communebe³chatów communebelskduży communebarlinek"
    test_sentence = train_sentences[1]
    print(loaded_vectorizer(test_sentence))
    # print(text_vectorizer(test_sentence))
    
        model_tokenizer = tf.keras.models.Sequential()
    model_tokenizer.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_tokenizer.add(text_vectorizer)

    # Create  tokenizer model.
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    output = loaded_vectorizer(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="text_model")

    model.summary()

    result=model.predict("bodykombimpv communebiałystok communebiałobrzegi communebe³¿yce communebe³chatów communebelskduży communebarlinek")
    print(result)




    """






