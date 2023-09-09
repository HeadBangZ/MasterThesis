import tensorflow as tf

nodes = 20
length = 5
features = 20
classes = 10

layers = [
    tf.keras.layers.SimpleRNN,
    tf.keras.layers.LSTM,
    tf.keras.layers.GRU
]

for lt in layers:
    model = tf.keras.models.Sequential([
        lt(nodes, activation='relu', input_shape=(length, features)),
        tf.keras.layers.Dense(classes, activation='relu')
    ])

    model.summary()
