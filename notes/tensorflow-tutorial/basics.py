import tensorflow as tf

# constants ####################################################################
a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

sess = tf.Session()

file_written = tf.summary.FileWriter('C:\\repos\\neureal-prototype-testing\\notes\\tensorflow-tutorial\\graph', sess.graph)
print(sess.run(c))

sess.close()

# cmd: tensorboard --logdir="TensorFlow"


# placeholders #################################################################

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

with tf.Session() as sess:
    output = sess.run( adder_node, {a : [1,3],
                                    b : [2,4]} )
    print(output)


# variables ####################################################################

# create model ##

# model parameters
W = tf.Variable([.3], tf.float32)       # if  W = -1
b = tf.Variable([-.3], tf.float32)      # and b =  1 then immediately converge.

# inputs and outputs
x = tf.placeholder(tf.float32)

# output values
y = tf.placeholder(tf.float32)
linear_model = W * x + b


# calculate loss, update variables.
squared_deltas= tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #print(sess.run( loss, { x:[1,2,3,4], y:[0,-1,-2,-3] } )

    # train
    for i in range(1000):
        sess.run(train, {   x:[1,2,3,4],
                            y:[0,-1,-2,-3] } )
        output = sess.run([W,b])
        print(output)
