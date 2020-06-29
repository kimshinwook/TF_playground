import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)
np.random.seed(0)
batch_size = 50
epoch_num = 5

mnist = input_data.read_data_sets(train_dir='./mnist', one_hot=True)

tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='tf_x') / 255.0
image_layer = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='tf_y')

test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
num_train_image = len(mnist.train.images)
print('train image number = ', num_train_image)


def conv_layer(inputs, filters, name):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3,
                               strides=1, padding='same', activation=tf.nn.relu,
                               use_bias=True,
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                               bias_initializer=tf.zeros_initializer,
                               kernel_regularizer=tf.nn.l2_loss,
                               name=name)
        return net


def fc_layer(inputs, units, name):
    with tf.variable_scope(name):
        net = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu, name=name,
                              kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                              bias_initializer=tf.zeros_initializer,
                              kernel_regularizer=tf.nn.l2_loss)

        net = tf.layers.dropout(net, rate=0.5)

        return net


def max_pool(inputs, name):
    net = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='same', name=name)
    return net


def vgg16(inputs):
    with tf.variable_scope('vgg16_net'):
        net = conv_layer(inputs=inputs, filters=64, name='conv1-1')
        net = conv_layer(inputs=net, filters=64, name='conv1-2')
        net = max_pool(inputs=net, name='pool1')                        # 14 x 14 x 64

        net = conv_layer(inputs=net, filters=128, name='conv2-1')
        net = conv_layer(inputs=net, filters=128, name='conv2-2')
        net = max_pool(inputs=net, name='pool2')                        # 7 x 7 x 128

        net = conv_layer(inputs=net, filters=256, name='conv3-1')
        net = conv_layer(inputs=net, filters=256, name='conv3-2')
        net = conv_layer(inputs=net, filters=256, name='conv3-3')
        net = max_pool(inputs=net, name='pool3')                        # 4 x 4 x 256

        net = conv_layer(net, 512, 'conv4-1')
        net = conv_layer(net, 512, 'conv4-2')
        net = conv_layer(net, 512, 'conv4-3')
        net = max_pool(net, 'pool4')                                    # 2 x 2 x 512

        #net = conv_layer(net, 512, 'conv5-1')
        #net = conv_layer(net, 512, 'conv5-2')
        #net = conv_layer(net, 512, 'conv5-3')
        #net = max_pool(net, 'pool5')

        net = tf.contrib.layers.flatten(net)
        net = fc_layer(net, 4096, name='fc1')
        net = fc_layer(net, 4096, name='fc2')

        output = tf.layers.dense(net, 10, name='logits')

        return output


output_logits = vgg16(image_layer)
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output_logits)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1),
                               predictions=tf.argmax(output_logits, axis=1), )[1]

sess = tf.Session()
init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
sess.run(init_op)

cur_position = 0
cur_epoch = 0
step = 0

while True:
    b_x, b_y = mnist.train.next_batch(batch_size)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    cur_position += batch_size
    step += 1
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, output_logits], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

    if cur_position >= num_train_image:
        cur_epoch += 1
        cur_position = 0
        print('## %d ecpoch finished' % cur_epoch)

    if cur_epoch == epoch_num:
        break

