import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer()) #W, b 를 사용하기 전에 반드시 초기화

for step in range(2001): #스텝 은 총2000번
    sess.run(train)
    if step % 100 == 0: #20번마다 출력
        print("step:{}\tcost:{:.5f}\tW:{}\tb:{}".format(step, sess.run(cost), sess.run(W), sess.run(b)))
