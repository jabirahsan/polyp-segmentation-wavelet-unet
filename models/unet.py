import tensorflow as tf


def conv_block(filters,inp):
    x=tf.keras.layers.Conv2D(filters,(3,3),padding='same',activation='relu')(inp)
    x=tf.keras.layers.BatchNormalization()(x)
    return x


def build_unet():
    """Builds the UNet Model
    Returns:
        UNet Model"""
    inp=tf.keras.layers.Input(shape=(256,256,3))
    x1=inp
    for _ in range(2):
        x1=conv_block(64,x1)
    
    x2=tf.keras.layers.MaxPooling2D((2,2))(x1)
    for _ in range(2):
        x2=conv_block(128,x2)

    x3=tf.keras.layers.MaxPooling2D((2,2))(x2)
    for _ in range(2):
        x3=conv_block(256,x3)

    x4=tf.keras.layers.MaxPooling2D((2,2))(x3)
    for _ in range(2):
        x4=conv_block(512,x4)

    bridge=tf.keras.layers.MaxPooling2D((2,2))(x4)
    bridge=conv_block(1024,bridge)

    up1=tf.keras.layers.UpSampling2D((2,2))(bridge)
    up1=tf.keras.layers.Concatenate()([up1,x4])
    up1=conv_block(1024,up1)
    up1=conv_block(512,up1)

    up2=tf.keras.layers.UpSampling2D((2,2))(up1)
    up2=tf.keras.layers.Concatenate()([up2,x3])
    up2=conv_block(512,up2)
    up2=conv_block(256,up2)

    up3=tf.keras.layers.UpSampling2D((2,2))(up2)
    up3=tf.keras.layers.Concatenate()([up3,x2])
    up3=conv_block(256,up3)
    up3=conv_block(128,up3)

    up4=tf.keras.layers.UpSampling2D((2,2))(up3)
    up4=tf.keras.layers.Concatenate()([up4,x1])
    up4=conv_block(128,up4)
    for _ in range(2):
        up4=conv_block(64,up4)
    output=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(up4)

    model=tf.keras.Model(inputs=inp,outputs=output)
    return model