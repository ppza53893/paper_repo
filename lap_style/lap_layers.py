import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.applications import VGG19


class AdaIN(Layer):
    """adaptive_instance_normalization.

    Args:
        content_feat (Tensor): Tensor with shape (N, H, W, C).
        style_feat (Tensor): Tensor with shape (N, H, W, C).

    Return:
        Normalized content_feat with shape (N, H, W, C)
    """

    def call(self, inputs):
        with tf.name_scope('op_adaIN'):
            style_mean = tf.math.reduce_mean(inputs[1], [1,2], keepdims=True)
            content_mean = tf.math.reduce_mean(inputs[0], [1,2], keepdims=True)

            style_std = tf.add(tf.math.reduce_std(inputs[1], [1,2], keepdims=True), tf.constant(1e-05))
            content_std = tf.add(tf.math.reduce_std(inputs[0], [1,2], keepdims=True), tf.constant(1e-05))

            normalized_feat = tf.divide(tf.subtract(inputs[0], content_mean), content_std)
            return tf.add(tf.multiply(normalized_feat, style_std), style_mean)


class ReflectionPadding2D(Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(int): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding_size = 1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        padding_width = padding_height = padding_size
        self.padding_tensor = tf.convert_to_tensor([
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ])

    def call(self, input_tensor):
        return tf.pad(input_tensor, self.padding_tensor, mode="REFLECT")


class ResnetBlock(Layer):
    """Residual block.

    It has a style of:
        ---Conv-ReLU-Conv-+-\\
         |________________|

    Args:
        dim (int): Channel number of intermediate features.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.ref_pad = ReflectionPadding2D()
        self.conv_1 = Conv2D(
            filters = self.dim,
            kernel_size = 3,
            kernel_initializer='he_normal',
            activation='relu')
        self.conv_2 = Conv2D(
            filters = self.dim,
            kernel_size = 3)
    
    def call(self, inputs):
        residual = self.ref_pad(inputs)
        residual = self.conv_1(residual)
        residual = self.ref_pad(residual)
        residual = self.conv_2(residual)

        return Add()([inputs, residual])


class LayerList(Layer):
    """paddle.nn.Sequential"""
    def __init__(self, layerlist: list, **kwargs):
        super().__init__(**kwargs)
        self.layerlist = layerlist
    
    def call(self, inputs):
        x = inputs
        for layer in self.layerlist:
            x = layer(x)
        return x


class RevisionNet(Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """

    def build(self, input_shape):
        self.DownBlock = LayerList([
            # Downblock #1
            ReflectionPadding2D(),
            Conv2D(
                filters=64,
                kernel_size=3,
                kernel_initializer='he_normal',
                activation='relu'
            ),
            # Downblock #2
            ReflectionPadding2D(),
            Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                kernel_initializer='he_normal',
                activation='relu'
            )
        ], name='r1')

        self.resblock = ResnetBlock(64, name='r2')

        self.UpBlock = LayerList([
            # Upblock #1
            UpSampling2D(),
            ReflectionPadding2D(),
            Conv2D(
                filters=64,
                kernel_size=3,
                kernel_initializer='he_normal',
                activation='relu'
            ),
            # Upblock #2
            UpSampling2D(),
            ReflectionPadding2D(),
            Conv2D(
                filters=3,
                kernel_size=3
            )
        ], name='r3')

    def call(self, inputs):
        out = self.DownBlock(inputs)
        out = self.resblock(out)
        out = self.UpBlock(out)
        return out


def get_vgg19_encoder():
    output_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    try:
        # online
        enc = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(128,128,3))
    except:
        # offline
        h5_path = os.path.join(
            os.path.split(
            os.path.abspath(__file__))[0],
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        if not os.path.exists(h5_path):
            raise FileNotFoundError()
        enc = VGG19(
            include_top=False,
            weights=h5_path,
            input_shape=(128,128,3))
    outputs=[enc.get_layer(layer_name).output for layer_name in output_layers]
    model = Model(
        inputs=enc.input,
        outputs=outputs,
        name = 'vgg19')
    model.trainable = False
    return model


def get_decoder():
    inputs = []
    for i in range(3):
        sm = 2**i
        inputs += [
                Input(shape=(16*sm, 16*sm, 512//sm), name=f'vgg_output1_{3-i}'), 
                Input(shape=(16*sm, 16*sm, 512//sm), name=f'vgg_output2_{3-i}')]

    x = AdaIN()([inputs[0], inputs[1]])
    x = ResnetBlock(512)(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(256, 3, kernel_initializer='he_normal', activation='relu')(x)

    x = UpSampling2D()(x)
    y = AdaIN()([inputs[2], inputs[3]])
    x = Add()([x, y])
    x = ResnetBlock(256)(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu')(x)

    x = UpSampling2D()(x)
    y = AdaIN()([inputs[4], inputs[5]])
    x = Add()([x, y])
    x = ReflectionPadding2D()(x)
    x = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu')(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu')(x)

    x = UpSampling2D()(x)
    x = ReflectionPadding2D()(x)
    x = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu')(x)
    x = ReflectionPadding2D()(x)
    output = Conv2D(3, 3)(x)

    model = Model(inputs, output, name='draft_decoder')
    return model
