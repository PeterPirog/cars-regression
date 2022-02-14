train_filepaths = 'G:\PycharmProject\cars-regression\datasets\*.csv'
train_filepaths = 'abalone_train.csv'
import time
import tensorflow as tf
import pandas as pd


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=(1,), dtype=tf.int64),
            args=(num_samples,)
        )


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            pass
            # Performing a training step
            # time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)


if __name__ == '__main__':
    df = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight", "Age"])

    # filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
    # print(filepath_dataset)

    print(df.head())
    columns = df.columns
    dataset = tf.data.experimental.make_csv_dataset('abalone_train.csv',
                                                    batch_size=3,
                                                    column_names=columns,
                                                    column_defaults=None,
                                                    label_name='Age',
                                                    na_value='',
                                                    shuffle=True,
                                                    shuffle_seed=42,
                                                    num_parallel_reads=tf.data.AUTOTUNE)
    #dataset = dataset.map(lambda x: x + 2)
    for line in dataset.take(3):
        print(line)

    print(dir(dataset))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam())

    model.fit(dataset, epochs=10)

    """

    dataset=tf.data.experimental.make_csv_dataset(
        file_pattern='abalone_train.csv', batch_size=5, column_names=columns, column_defaults=None,
        label_name=None, select_columns=None, field_delim=';',
        use_quote_delim=True, na_value='', header=True, num_epochs=None,
        shuffle=True, shuffle_buffer_size=10000, shuffle_seed=42,
        prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
        num_rows_for_inference=100, compression_type=None, ignore_errors=True
    )



    

    n_readers = 3
    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers,
        num_parallel_calls=tf.data.AUTOTUNE)

    print(tf.data.AUTOTUNE)
    for line in dataset.take(5):
        print(line)

    #Execution time: 151.0507352 BASIC
    #benchmark(dataset.prefetch(tf.data.AUTOTUNE) Execution time: 147.047525

    benchmark(dataset.prefetch(tf.data.AUTOTUNE)
)
    tf.data.experimental.make_csv_dataset
    """
