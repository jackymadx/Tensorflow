#Operation Constant
import tensorflow as tf

a = tf.constant(5,name="a")
b = tf.constant(15,name="b")
b2 = tf.constant(2,name="b2")
c = tf.add(a,b,name="c")
d = tf.add(a,b,name="d")
e = tf.subtract(a,b2,name="e")

sess = tf.Session()
output = sess.run(c);
print(output)
output2 = sess.run(e);
print(output2)
sess.close()
