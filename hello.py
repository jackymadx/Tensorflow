#Import tensor
import tensorflow as tf

#Create tensor constant
hello = tf.constant('Hello, TensorFlow!')

#Start tf session
sess = tf.Session()

#Run session
print(sess.run(hello)) 
