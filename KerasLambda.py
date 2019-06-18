import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

reparametrisation = True  # Change for control model

data = tf.keras.datasets.mnist  # Change for CIFAR-10 dataset
data_shape = (28, 28)  # Change to (32, 32, 3) for CIFAR-10 dataset

num_sample = 1000

(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


def reparam(tensors):
    loc = tensors[0]
    scale = tensors[1]
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    return tf.transpose(dist.sample([num_sample], name="logits"), [1, 0, 2])


inputs = tf.keras.layers.Input(shape=data_shape)
flatten = tf.keras.layers.Flatten()(inputs)
dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flatten)

if reparametrisation:
    
    locs = tf.keras.layers.Dense(units=10, name="means")(dense)
    
    scales = tf.keras.layers.Dense(units=10, name="std_devs",
                               activation=tf.nn.softplus)(dense)
    
    repar = tf.keras.layers.Lambda(reparam, name="normal")([locs, scales])
    
    out_reparam = tf.keras.layers.Dense(10, name="out", activation=tf.nn.softmax)(repar)

    model_reparam = tf.keras.models.Model(inputs=inputs, outputs=out_reparam)
    
    model_reparam.compile(optimizer='adam',
                          loss='categorical_crossentropy', metrics=['acc'])

    model_reparam.summary()
    tf.keras.utils.plot_model(model_reparam, "model_reparam.png")

    y_train_tiled = np.tile(y_train[:, tf.newaxis, :], [1, num_sample, 1])
    y_test_tiled = np.tile(y_test[:, tf.newaxis, :], [1, num_sample, 1])

    model_reparam.fit(x_train, y_train_tiled, epochs=5)
    model_reparam.evaluate(x_test, y_test_tiled)

else:
    out_no_reparam = tf.keras.layers.Dense(10, name="out", activation=tf.nn.softmax)(dense)

    model_no_reparam = tf.keras.models.Model(inputs=inputs, outputs=out_no_reparam)
    model_no_reparam.compile(optimizer='adam',
                             loss='categorical_crossentropy', metrics=['acc'])

    model_no_reparam.summary()
    tf.keras.utils.plot_model(model_no_reparam, "model_no_reparam.png")

    model_no_reparam.fit(x_train, y_train, epochs=5)
    model_no_reparam.evaluate(x_test, y_test)
