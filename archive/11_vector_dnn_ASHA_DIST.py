# UBUNTU
import numpy as np
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.suggest.hyperopt import HyperOptSearch
from tools.general_tools import nll_loss

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def dist_2_mu_std(dist, invert_log10=True):
    mu = dist.mean().numpy()
    sigma = dist.stddev().numpy()

    if invert_log10:
        val_max = tf.pow(10.0, mu + sigma)
        mu = tf.pow(10.0, mu)
        sigma = val_max - mu
    return mu, sigma

def nll_mape_metric(y_true, dist, k=2.0):
    mu, sigma = dist_2_mu_std(dist, invert_log10=True)
    y_true = tf.pow(10.0, y_true)
    mape = 100 * k * sigma / y_true
    metric = tf.reduce_mean(mape, axis=0)
    return metric

def NLL(y, distr):
  return -distr.log_prob(y)

def my_dist(params):
  return tfd.Normal(loc=params[:,0:1],
                    #scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable
                    scale=tf.math.exp(params[:,1:2]))# both parameters are learnable

def load_data(N=None):
    embeded_train_sentences= np.load('/home/ppirog/projects/cars-regression/npdata/embeded_train_sentences.npy')
    embeded_val_sentences = np.load('/home/ppirog/projects/cars-regression/npdata/embeded_val_sentences.npy')

    train_labels = np.load('/home/ppirog/projects/cars-regression/npdata/train_labels.npy')
    val_labels = np.load('/home/ppirog/projects/cars-regression/npdata/val_labels.npy')
    if N is None:
        return embeded_train_sentences,embeded_val_sentences,train_labels,val_labels
    else:
        return embeded_train_sentences[:N,:], embeded_val_sentences[:N,:], train_labels[:N], val_labels[:N]

def train_net(config):


    X_train, X_test, y_train, y_test = load_data(N=10000)

    epochs = 5000
    # define model
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]), dtype=tf.float32)

    x = tf.keras.layers.Flatten()(inputs)
    #x = tf.keras.layers.LayerNormalization()(x)

    for i in range(config["hidden_layers"]):
        x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                                  activation=config["activation"])(x)
        #x = tf.keras.layers.Dropout(config["dropout1"])(x)

    out_mean= tf.keras.layers.Dense(1, activation="linear")(x)
    out_std= tf.keras.layers.Dense(1, activation="linear")(x)
    params = tf.keras.layers.Concatenate()([out_mean, out_std])

    #params = tf.keras.layers.Dense(2, activation="linear")(x)
    dist = tfp.layers.DistributionLambda(my_dist)(params)



    model = tf.keras.Model(inputs=inputs, outputs=dist, name="dist_model")

    model.compile(loss=NLL,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
                #  metrics=[NLL]) #nll_mape_metric

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
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

    ray.init(dashboard_port=12302)

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
        name="dist_model",
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
        num_samples=100,  # number of samples from hyperparameter space
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
            "batch": tune.choice([32]),
            #"noise_std": tune.uniform(0.01, 0.4),
            "learning_rate": tune.choice([0.001]),
            # Layer 1 params
            "hidden1": tune.randint(3, 200),
            "activation": tune.choice(["elu"]),
            #"dropout1": tune.uniform(0.01, 0.15),
            "hidden_layers": tune.randint(1, 4)
        }

    )
    print("Best result:", analysis.best_result, "Best hyperparameters found were: ", analysis.best_config)
    logs='/home/ppirog/projects/cars-regression/ray_results'
    # python -m tensorboard.main --logdir logs --bind_all --port=12301
    # python -m tensorboard.main --logdir /home/ppirog/projects/cars-regression/ray_results --bind_all --port=12301