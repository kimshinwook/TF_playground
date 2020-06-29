import tensorflow as tf

var = tf.Variable(0)    # our first variable in the "global_variable" set

add_rst = tf.add(var, 1)
update_value = tf.assign(var, add_rst)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(update_value)
        print(sess.run(var))