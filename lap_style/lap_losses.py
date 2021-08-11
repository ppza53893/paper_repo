from typing import Union

import numpy as np
import tensorflow as tf


def calc_mean_std(feat, eps: float=1e-05):
    mean = tf.math.reduce_mean(feat, axis=[1,2], keepdims=True)
    std = tf.math.reduce_std(feat, axis=[1,2], keepdims=True) + eps
    return mean, std


def calc_variance_norm(feat: tf.Tensor):
    """mean_variance_norm.

    Args:
        feat (tf.Tensor): Tensor with shape (N, H, W, C).

    Return:
        Normalized feat with shape (N, H, W, C)
    """
    mean, std = calc_mean_std(feat)
    return tf.math.divide_no_nan((feat-mean), std)


def calc_emb_loss(y_true, y_pred):
    yt_norm = tf.math.l2_normalize(y_true, axis=-1)
    yp_norm = tf.math.l2_normalize(y_pred, axis=-1)
    return 1. - tf.reduce_sum(yt_norm * yp_norm, axis=-1)


class Base:
    def call(self, y_true, y_pred):
        return self.__call__(y_true, y_pred)
    
    def __call__(self, y_true, y_pred):
        pass


class CalcStyleEmdLoss(Base):
    """Calc Style Emd Loss.
    """
    def __call__(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray]
        ):
        """Forward Function.

        Args:
            y_pred (tf.Tensor): of shape (N, H, W, C). Predicted tensor.
            y_true (tf.Tensor): of shape (N, H, W, C). Ground truth tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        CX_M = calc_emb_loss( y_true, y_pred)
        m1 = tf.reduce_mean(tf.reduce_min(CX_M, axis=2))
        m2 = tf.reduce_mean(tf.reduce_min(CX_M, axis=1))
        m = tf.stack([m1, m2])
        return tf.reduce_max(m)


class CalcContentReltLoss(Base):
    """Calc Content Relt Loss.
    """
    def __call__(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray]
        ):
        """Forward Function.

        Args:
            y_pred (tf.Tensor): of shape (N, H, W, C). Predicted tensor.
            y_true (tf.Tensor): of shape (N, H, W, C). Ground truth tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        dM = 1.
        Mx = calc_emb_loss(y_pred, y_pred)
        Mx = tf.math.divide_no_nan(Mx, tf.reduce_sum(Mx, axis=1, keepdims=True))
        My = calc_emb_loss(y_true, y_true)
        My = tf.math.divide_no_nan(My, tf.reduce_sum(My, axis=1, keepdims=True))
        loss_content = tf.reduce_mean(tf.abs(dM * (Mx - My))) * y_pred.shape[1] * y_pred.shape[2]
        return loss_content


class CalcContentLoss(Base):
    """Calc Content Loss.
    """
    def __init__(self, norm: bool=False):
        self.mse = tf.keras.losses.MeanSquaredError()
        self.norm = norm

    def __call__(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray],
        ):
        """Forward Function.

        Args:
            y_pred (tf.Tensor): of shape (N, H, W, C). Predicted tensor.
            y_true (tf.Tensor): of shape (N, H, W, C). Ground truth tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if not self.norm:
            loss =  self.mse(y_true, y_pred)
        else:
            loss = self.mse(
                calc_variance_norm(y_true),
                calc_variance_norm(y_pred)
            )
        return loss


class CalcStyleLoss(Base):
    """Calc Style Loss.
    """
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()

    def __call__(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray]
        ):
        """
        Args:
            y_pred (tf.Tensor): of shape (N, H, W, C). Predicted tensor.
            y_true (tf.Tensor): of shape (N, H, W, C). Ground truth tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        pred_mean, pred_std = calc_mean_std(y_pred)
        target_mean, target_std = calc_mean_std(y_true)
        loss = self.mse(target_mean, pred_mean) + self.mse(target_std, pred_std)
        return loss


def content_w_range(
    yt_1, yt_2, yt_3, yt_4, yt_5, yp_1, yp_2, yp_3, yp_4, yp_5
):
    loss_fn = CalcContentLoss(norm=True)
    loss = loss_fn(yt_1, yp_1)
    loss += loss_fn(yt_2, yp_2)
    loss += loss_fn(yt_3, yp_3)
    loss += loss_fn(yt_4, yp_4)
    loss += loss_fn(yt_5, yp_5)
    return loss


def content_wo_range(y_true, y_pred):
    return CalcContentLoss()(y_true, y_pred)


def style_w_range(
    yt_1, yt_2, yt_3, yt_4, yt_5, yp_1, yp_2, yp_3, yp_4, yp_5
):
    loss_fn =  CalcStyleLoss()
    loss = loss_fn(yt_1, yp_1)
    loss += loss_fn(yt_2, yp_2)
    loss += loss_fn(yt_3, yp_3)
    loss += loss_fn(yt_4, yp_4)
    loss += loss_fn(yt_5, yp_5)
    return loss


def style_remd(
    yt_3, yt_4, yp_3, yp_4
):
    loss_fn = CalcStyleEmdLoss()
    return loss_fn(yt_3, yp_3) + loss_fn(yt_4, yp_4)


def content_relt(
    yt_3, yt_4, yp_3, yp_4
):
    loss_fn = CalcContentReltLoss()
    return loss_fn(yt_3, yp_3) + loss_fn(yt_4, yp_4)


def prepare_draft_losses():
    content_loss = content_w_range
    style_loss = style_w_range
    identity_loss_1 = CalcContentLoss().call
    identity_loss_2 = content_w_range
    style_remd_loss = style_remd
    content_relt_loss = content_relt

    return content_loss, style_loss, identity_loss_1, identity_loss_2, style_remd_loss, content_relt_loss


