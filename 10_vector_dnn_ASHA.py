# UBUNTU
import numpy as np
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.suggest.hyperopt import HyperOptSearch


def load_data():
    embeded_train_sentences= np.load('/home/ppirog/projects/cars-regression/npdata/embeded_train_sentences.npy')
    embeded_val_sentences = np.load('/home/ppirog/projects/cars-regression/npdata/embeded_val_sentences.npy')

    train_labels = np.load('/home/ppirog/projects/cars-regression/npdata/train_labels.npy')
    val_labels = np.load('/home/ppirog/projects/cars-regression/npdata/val_labels.npy')
    return embeded_train_sentences,embeded_val_sentences,train_labels,val_labels

def train_net(config):


    X_train, X_test, y_train, y_test = load_data()

    epochs = 5
    # define model
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]), dtype=tf.float32)

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=config["noise_std"])(x)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation"])(x)
    # x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation"])(x)
    # x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error

    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=20),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=10),
                      # tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                      #                                  monitor='val_loss',
                      #                                 save_best_only=True),
                      TuneReportCallback({'val_loss': 'val_loss'})]

    model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)


if __name__ == "__main__":
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

    sched_asha = ASHAScheduler(time_attr="training_iteration",
                               max_t=500,
                               grace_period=10,
                               # mode='max', #find maximum, do not define here if you define in tune.run
                               reduction_factor=3,
                               # brackets=1
                               )

    analysis = tune.run(
        train_net,
        search_alg=HyperOptSearch(),
        name="keras2",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        metric="val_loss",
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 500
        },
        num_samples=10,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/ppirog/projects/cars-regression/ray_results',
        # default value is ~/ray_results /root/ray_results/
        resources_per_trial={
            "cpu": 16,
            "gpu": 0
        },
        config={
            # training parameters
            "batch": tune.choice([128]),
            "noise_std": tune.uniform(0.01, 0.4),
            "learning_rate": tune.choice([0.001]),
            # Layer 1 params
            "hidden1": tune.randint(3, 10),
            "activation": tune.choice(["elu"]),
            "dropout1": tune.uniform(0.01, 0.15),
            # Layer 2 params
            "hidden2": tune.randint(3, 10),
            "dropout2": tune.uniform(0.01, 0.15),  # tune.choice([0.01, 0.02, 0.05, 0.1, 0.2])

        }

    )
    print("Best result:", analysis.best_result, "Best hyperparameters found were: ", analysis.best_config)