import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os

train_x = np.linspace(-5, 3, 50)
train_y = train_x * 5 + 10 + np.random.random(50) * 10 - 5

plt.plot(train_x, train_y, 'r.')
plt.grid(True)
plt.show()

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.random.truncated_normal([1]), name='Weight')
b = tf.Variable(tf.random.truncated_normal([1]), name='bias')

z = tf.multiply(X, w) + b

cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_epochs = 20
display_step = 2


saver = tf.train.Saver(max_to_keep=15)
savedir = "model/"


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        loss_list = []
        for epoch in range(training_epochs):
            for (x, y) in zip(train_x, train_y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: x, Y: y})
                loss_list.append(loss)
                print('Iter: ', epoch, ' Loss: ', loss)

            w_, b_ = sess.run([w, b], feed_dict={X: x, Y: y})

            saver.save(sess, savedir + "linear.cpkt", global_step=epoch)

        print(" Finished ")
        print("W: ", w_, " b: ", b_, " loss: ", loss)
        plt.plot(train_x, train_x * w_ + b_, 'g-', train_x, train_y, 'r.')
        plt.grid(True)
        plt.show()

    load_epoch = 10

    with tf.Session() as sess2:
        sess2.run(tf.global_variables_initializer())
        saver.restore(sess2, savedir + "linear.cpkt-" + str(load_epoch))
        print(sess2.run([w, b], feed_dict={X: train_x, Y: train_y}))