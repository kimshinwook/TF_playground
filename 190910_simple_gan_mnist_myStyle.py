import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets(train_dir='./mnist', one_hot=True)

# Generator network design
with tf.variable_scope('Generator'):
    G_in = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    G_l0 = tf.layers.dense(inputs=G_in, units=256, activation=tf.nn.relu)
    G_out = tf.layers.dense(inputs=G_l0, units=28*28, activation=tf.nn.sigmoid)

with tf.variable_scope('Discriminator'):
    D_in = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
    D_l0 = tf.layers.dense(inputs=D_in, units=256, activation=tf.nn.relu, name='l')
    D_out0 = tf.layers.dense(inputs=D_l0, units=1, activation=tf.nn.sigmoid, name='out')

    D_l1 = tf.layers.dense(inputs=G_out, units=256, activation=tf.nn.relu, name='l', reuse=True)
    D_out1 = tf.layers.dense(inputs=D_l1, units=1, activation=tf.nn.sigmoid, name='out', reuse=True)


loss_D = -tf.reduce_mean(tf.log(D_out0) + tf.log(1-D_out1))
loss_G = -tf.reduce_mean(tf.log(D_out1))
opt_D = tf.train.AdamOptimizer(learning_rate=0.0002)
opt_G = tf.train.AdamOptimizer(learning_rate=0.0002)
train_D = opt_D.minimize(loss_D, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Discriminator'))
train_G = opt_G.minimize(loss_G, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

noise_test = np.random.normal(size=(10, 128))
for epoch in range(200):  # 200 = Num. of Epoch
    for i in range(int(mnist.train.num_examples / 100)):  # 100 = Batch Size
        batch_xs, _ = mnist.train.next_batch(100)
        noise = np.random.normal(size=(100, 128))

        sess.run(train_D, feed_dict={D_in: batch_xs, G_in: noise})
        sess.run(train_G, feed_dict={G_in: noise})

    if epoch == 0 or (epoch + 1) % 10 == 0:  # 10 = Saving Period
        samples = sess.run(G_out, feed_dict={G_in: noise_test})

        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig('samples_ex/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)