import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Random seed initialize
np.random.seed(0)
tf.set_random_seed(0)

# Ground-truth generating
x = np.linspace(-1, 1, 100)  # shape = (100, )
x = x[:, np.newaxis]  # shape = (100, 1) this shape can be input to the graph

noise = np.random.normal(0, 0.1, size=x.shape)  # random noise generated

y = np.power(x, 2) + noise  # output with additive noise

# TF graph generating
tf_x = tf.placeholder(dtype=tf.float32, shape=x.shape)
tf_y = tf.placeholder(dtype=tf.float32, shape=y.shape)

layer_1 = tf.layers.dense(inputs=tf_x, units=20, activation=tf.nn.relu)
output = tf.layers.dense(inputs=layer_1, units=1)

# Define loss and optimizer
loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

plt.ion()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        pred, l, _ = sess.run([output, loss, train_op], feed_dict={tf_x: x, tf_y: y})

        if (i % 5) == 0:
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x, pred, 'r-', lw=5)
            plt.text(0.3, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

plt.ioff()
plt.show()
