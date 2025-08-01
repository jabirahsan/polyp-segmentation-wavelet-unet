import tensorflow as tf

"""Imported From: https://idiotdeveloper.com/resunet-implementation-in-tensorflow/"""

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = tf.keras.layers.Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = tf.keras.layers.Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = tf.keras.layers.BatchNormalization()(s)

    ## Add
    x = tf.keras.layers.Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = tf.keras.layers.Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = tf.keras.layers.BatchNormalization()(s)

    ## Add
    x = tf.keras.layers.Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = tf.keras.layers.Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)

    x2 = tf.keras.layers.Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = tf.keras.layers.BatchNormalization()(x2)

    x3 = tf.keras.layers.Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = tf.keras.layers.BatchNormalization()(x3)

    x4 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = tf.keras.layers.BatchNormalization()(x4)

    y = tf.keras.layers.Add()([x1, x2, x3, x4])
    y = tf.keras.layers.Conv2D(num_filters, (1, 1), padding="same")(y)
    return y

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = tf.keras.layers.BatchNormalization()(g)
    g_conv = tf.keras.layers.Activation("relu")(g_conv)
    g_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = tf.keras.layers.BatchNormalization()(x)
    x_conv = tf.keras.layers.Activation("relu")(x_conv)
    x_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = tf.keras.layers.Add()([g_pool, x_conv])

    gc_conv = tf.keras.layers.BatchNormalization()(gc_sum)
    gc_conv = tf.keras.layers.Activation("relu")(gc_conv)
    gc_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = tf.keras.layers.Multiply()([gc_conv, x])
    return gc_mul

class ResUnetPlusPlus:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        n_filters = [16, 32, 64, 128, 256]
        inputs = tf.keras.layers.Input((self.input_size, self.input_size, 3))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = tf.keras.layers.UpSampling2D((2, 2))(d1)
        d1 = tf.keras.layers.Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = tf.keras.layers.UpSampling2D((2, 2))(d2)
        d2 = tf.keras.layers.Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = tf.keras.layers.UpSampling2D((2, 2))(d3)
        d3 = tf.keras.layers.Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = tf.keras.layers.Activation("sigmoid")(outputs)

        ## Model
        model = tf.keras.Model(inputs, outputs)
        return model