import tensorflow as tf

#Placeholder 버젼
print("\n\nPlaceHolder Example")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = tf.add(tf.matmul(W,X), b)

cost_val = tf.reduce_mean(tf.square(h - Y))

for step in range(2001):

