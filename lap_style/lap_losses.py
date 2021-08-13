import tensorflow as tf


__all__ = [
    "content_loss",
    "content_relt_loss",
    "identity_loss",
    "style_loss",
    "style_remd_loss"
]


MSE = tf.keras.losses.MeanSquaredError()


@tf.function(experimental_relax_shapes=True)
def content_loss(y_true, y_pred):
    with tf.name_scope('op_content'):
        yt_m = tf.math.reduce_mean(y_true, axis=[1,2], keepdims=True)
        yp_m = tf.math.reduce_mean(y_pred, axis=[1,2], keepdims=True)

        yt_std = tf.math.reduce_std(y_true, axis=[1,2], keepdims=True)
        yp_std = tf.math.reduce_std(y_pred, axis=[1,2], keepdims=True)

        yt_ns = tf.divide(tf.subtract(y_true, yt_m), yt_std+tf.constant(1e-05))
        yp_ns = tf.divide(tf.subtract(y_pred, yp_m), yp_std+tf.constant(1e-05))
        return MSE(yt_ns, yp_ns)


@tf.function(experimental_relax_shapes=True)
def identity_loss(y_true, y_pred):
    with tf.name_scope('op_identity1'):
        return MSE(y_true, y_pred)


@tf.function(experimental_relax_shapes=True)
def style_loss(y_true, y_pred):
    with tf.name_scope('op_style'):
        yt_m = tf.math.reduce_mean(y_true, axis=[1,2], keepdims=True)
        yp_m = tf.math.reduce_mean(y_pred, axis=[1,2], keepdims=True)

        yt_std = tf.math.reduce_std(y_true, axis=[1,2], keepdims=True)
        yp_std = tf.math.reduce_std(y_pred, axis=[1,2], keepdims=True)

        return tf.add(MSE(yt_m, yp_m), MSE(yt_std, yp_std))


@tf.function(experimental_relax_shapes=True)
def content_relt_loss(y_true, y_pred):
    with tf.name_scope('op_content_relative'):
        # square
        yt_square = tf.math.square(y_true)
        yp_square = tf.math.square(y_pred)

        # calc
        yt_ns = tf.math.divide_no_nan(yt_square, tf.reduce_sum(yt_square, axis=-1, keepdims=True))
        yp_ns = tf.math.divide_no_nan(yp_square, tf.reduce_sum(yp_square, axis=-1, keepdims=True))

        # distance
        yt_dist = tf.subtract(tf.constant(1.0), tf.reduce_sum(yt_ns, axis=-1))
        yp_dist = tf.subtract(tf.constant(1.0), tf.reduce_sum(yp_ns, axis=-1))

        # divide
        yt_mx = tf.math.divide_no_nan(yt_dist, tf.reduce_sum(yt_dist, axis=-1, keepdims=True))
        yp_mx = tf.math.divide_no_nan(yt_dist, tf.reduce_sum(yp_dist, axis=-1, keepdims=True))

        # avg, multiply
        mxy = tf.reduce_mean(tf.abs(tf.subtract(yt_mx, yp_mx)))
        mul_var = tf.convert_to_tensor(y_true.shape[1]*y_true.shape[2], tf.float32)
        return tf.multiply(mxy, mul_var)


@tf.function(experimental_relax_shapes=True)
def style_remd_loss(y_true, y_pred):
    with tf.name_scope('op_style_rEMD'):
        # l2
        yt_l2 = tf.math.l2_normalize(y_true, axis=-1)
        yp_l2 = tf.math.l2_normalize(y_pred, axis=-1)

        # dot
        ytp = tf.einsum('ijkl,ijkl->ijk', yt_l2, yp_l2)
        emb = tf.subtract(tf.constant(1.0), ytp)

        m1 = tf.reduce_mean(tf.reduce_min(emb, axis=1))
        m2 = tf.reduce_mean(tf.reduce_min(emb, axis=2))

        return tf.reduce_max(tf.stack([m1, m2]))
