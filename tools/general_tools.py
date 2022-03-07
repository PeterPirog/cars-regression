import os
import io
import numpy as np
import tensorflow as tf
import math
from matplotlib import pyplot as plt

try:
    import ray
    ray.init()
    import modin.pandas as pd

except:
    import pandas as pd

pd.set_option('display.max_columns', None)


# function to check if path to file or directory exist
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def dataframe_analysis(df, xls_filename='Columns_analysis.xlsx'):
    # Delete old analysis file if exist
    if os.path.exists(xls_filename):
        os.remove(xls_filename)

    # Analysis of  unique values
    output = []

    for col in df.columns:
        nonNull = len(df) - np.sum(pd.isna(df[col]))
        unique = df[col].nunique()
        colType = str(df[col].dtype)

        output.append([col, nonNull, unique, colType])

    output = pd.DataFrame(output)
    output.columns = ['colName', 'non-null values', 'unique', 'dtype']
    output = output.sort_values(by='unique', ascending=False)
    output.to_excel(xls_filename, sheet_name='Columns')

    # Return categorical columns
    # get all categorical columns in the dataframe
    catCols = [col for col in df.columns if df[col].dtype == "O"]
    numCols = [col for col in df.columns if not df[col].dtype == "O"]
    # print(output)
    return output, catCols, numCols

def nll_loss(y_true, y_pred):
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


def embedings2tsv_files(embed_weights_from_layer,words_in_vocab,
                        embedding_vectors_file_name="embedding_vectors.tsv",
                        embedding_metadata_file_name="embedding_metadata.tsv"):
    # Get the weight matrix of embedding layer
    # (these are the numerical patterns between the text in the training dataset the model has learned)
    embed_weights = embed_weights_from_layer
    #print(embed_weights.shape)  # same size as vocab size and embedding_dim (each word is a embedding_dim size vector)

    # # Code below is adapted from: https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_trained_word_embeddings_and_save_them_to_disk


    #words_in_vocab = loaded_vectorizer.get_vocabulary()
    # # Create output writers
    out_v = io.open(embedding_vectors_file_name, "w", encoding="utf-8")
    out_m = io.open(embedding_metadata_file_name, "w", encoding="utf-8")

    # # Write embedding vectors and words to file
    for num, word in enumerate(words_in_vocab):
        if num == 0:
            continue  # skip padding token
        vec = embed_weights[num]
        out_m.write(word + "\n")  # write words to file
        out_v.write("\t".join([str(x) for x in vec]) + "\n")  # write corresponding word vector to file
    out_v.close()
    out_m.close()
    print(f'Files: {embedding_vectors_file_name}, {embedding_metadata_file_name} are saved.')

def create_training_plots(history,metric_name='mse',loss_name='loss'):
    # Create mse and loss plots
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_'+metric_name])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history[loss_name])
    plt.plot(history.history['val_'+loss_name])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    result=dir_path('/home/ppirog/projects/cars-regression/filtered_dataset')
    print(result)