import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
print(tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
))
tf.test.gpu_device_name()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with tf.Session() as sess:
    print (sess.run(c))