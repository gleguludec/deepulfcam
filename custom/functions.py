import tensorflow as tf

tfs = tf.sparse

def psnr(y_true, y_pred):
    mse = tf.reduce_mean((y_true - y_pred)**2)
    return -10 * tf.math.log(mse) / tf.math.log(10.)