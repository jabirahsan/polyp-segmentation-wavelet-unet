import tensorflow as tf

def get_haar_wavelet_filters():
    # Haar wavelet filters for 2D DWT
    ll = tf.constant([[0.5, 0.5],
                      [0.5, 0.5]], dtype=tf.float32)
    lh = tf.constant([[0.5, 0.5],
                      [-0.5, -0.5]], dtype=tf.float32)
    hl = tf.constant([[0.5, -0.5],
                      [0.5, -0.5]], dtype=tf.float32)
    hh = tf.constant([[0.5, -0.5],
                      [-0.5, 0.5]], dtype=tf.float32)

    filters = tf.stack([ll, lh, hl, hh], axis=-1)  # shape: (2, 2, 4)
    filters = tf.reshape(filters, [2, 2, 1, 4])    # shape: (2, 2, in_channels, out_channels)
    return filters



class DWTPooling(tf.keras.layers.Layer):
    def __init__(self,components='LL',**kwargs):
        super(DWTPooling, self).__init__(**kwargs)
        self.filters = get_haar_wavelet_filters()
        self.components=components
    def call(self, x):
        B, H, W, C = x.shape

        outputs = []

        # Apply DWT to each channel
        for c in range(C):
            x_c = x[:, :, :, c:c+1]  # shape (B, H, W, 1)
            filtered = tf.nn.conv2d(x_c, self.filters, strides=2, padding='SAME')  # shape (B, H/2, W/2, 4)
            if self.components=='LL':
                out=filtered[:,:,:,0:1]
            if self.components=='LH':
                out=filtered[:,:,:,1:2]
            if self.components=='HL':
                out=filtered[:,:,:,2:3]
            if self.components=='HH':
                out=filtered[:,:,:,-1]
            else:
                out=filtered
                
            outputs.append(out)

        # Concatenate along channel axis
        return tf.concat(outputs, axis=-1)  # shape (B, H/2, W/2, C)




class InceptionEncoder(tf.keras.layers.Layer):
    def __init__(self, layer_filters, out_filters, name_prefix, wavelet_component='LL',**kwargs):
        super(InceptionEncoder, self).__init__(**kwargs)

        self.name_prefix = name_prefix
        self.layer_filters = layer_filters
        self.out_filters = out_filters

        # First path
        self.conv1 = tf.keras.layers.Conv2D(layer_filters, (1, 1), strides=2, activation='relu',
                                   padding='same', name=f"{name_prefix}_conv1x1")
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Second path: 1x1 → 3x3
        self.conv2a = tf.keras.layers.Conv2D(layer_filters, (1, 1), strides=2, activation='relu',
                                    padding='same', name=f"{name_prefix}_conv1x1_3x3_a")
        self.conv2b = tf.keras.layers.Conv2D(layer_filters, (3, 3), activation='relu',
                                    padding='same', name=f"{name_prefix}_conv1x1_3x3_b")
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Third path: 1x1 → 5x5
        self.conv3a = tf.keras.layers.Conv2D(layer_filters, (1, 1), strides=2, activation='relu',
                                    padding='same', name=f"{name_prefix}_conv1x1_5x5_a")
        self.conv3b = tf.keras.layers.Conv2D(layer_filters, (5, 5), activation='relu',
                                    padding='same', name=f"{name_prefix}_conv1x1_5x5_b")
        self.bn3 = tf.keras.layers.BatchNormalization()

        # Fourth path: Hybrid pooling → 1x1 conv
        self.hybrid_pool = DWTPooling(components=wavelet_component, name=f"{name_prefix}_hybrid_pooling")
        self.conv4 = tf.keras.layers.Conv2D(layer_filters, (1, 1), activation='relu',
                                   padding='same', name=f"{name_prefix}_pooling")

        # Concatenation and final projection
        self.concat = tf.keras.layers.Concatenate(name=f"{name_prefix}_concat")
        self.project = tf.keras.layers.Conv2D(out_filters, (1, 1), name=f"{name_prefix}_output")

    def call(self, inputs):
        # First path
        x1 = self.bn1(self.conv1(inputs))

        # Second path
        x2 = self.conv2a(inputs)
        x2 = self.conv2b(x2)
        x2 = self.bn2(x2)

        # Third path
        x3 = self.conv3a(inputs)
        x3 = self.conv3b(x3)
        x3 = self.bn3(x3)

        # Fourth path
        x4 = self.hybrid_pool(inputs)
        x4 = self.conv4(x4)

        # Concatenate and project
        x = self.concat([x1, x2, x3, x4])
        x = self.project(x)

        return x



class InceptionDecoder(tf.keras.layers.Layer):
    def __init__(self,layer_filters,out_filters,name_prefix,**kwargs):
        super(InceptionDecoder,self).__init__(**kwargs)
        self.name_prefix = name_prefix
        self.layer_filters = layer_filters
        self.out_filters = out_filters

        #First Path
        self.first=tf.keras.layers.Conv2D(layer_filters,(1,1),strides=2,activation='relu',name=f"{name_prefix}_conv1x1",padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()

        #Second path
        self.seconda=tf.keras.layers.Conv2D(layer_filters,(1,1),strides=2,activation='relu',name=f"{name_prefix}_conv1x1_3x3_a",padding='same')
        self.secondb=tf.keras.layers.Conv2D(layer_filters,(3,3),activation='relu',name=f"{name_prefix}_conv1x1_3x3_b",padding='same')
        self.bn2=tf.keras.layers.BatchNormalization()

        #Third Path
        self.thirda=tf.keras.layers.Conv2D(layer_filters,(1,1),strides=2,activation='relu',name=f"{name_prefix}_conv1x1_5x5_a",padding='same')
        self.thirdb=tf.keras.layers.Conv2D(layer_filters,(5,5),activation='relu',name=f"{name_prefix}_conv1x1_5x5_b",padding='same')
        self.bn3=tf.keras.layers.BatchNormalization()

        #Fourth Path
        self.fourtha=DWTPooling(components='LL', name=f"{name_prefix}_hybrid_pooling")
        self.fourthb=tf.keras.layers.Conv2D(layer_filters,(1,1),padding='same',activation='relu',name=f"{name_prefix}_pooling")

        #concat
        self.concat=tf.keras.layers.Concatenate(name=f"{name_prefix}_concat")
        self.project=tf.keras.layers.Conv2DTranspose(out_filters,(1,1),strides=2,name=f"{name_prefix}_output")

    def call(self,inputs):
        # First path
        x1 = self.bn1(self.first(inputs))

        # Second path
        x2 = self.seconda(inputs)
        x2 = self.secondb(x2)
        x2 = self.bn2(x2)

        # Third path
        x3 = self.thirda(inputs)
        x3 = self.thirdb(x3)
        x3 = self.bn3(x3)

        # Fourth path
        x4 = self.fourtha(inputs)
        x4 = self.fourthb(x4)

        # Concatenate and project
        x = self.concat([x1, x2, x3, x4])
        x = self.project(x)
        return x


class ASPP_Block(tf.keras.layers.Layer):
    def __init__(self,filters,name_prefix,rate=1,**kwargs):
        super(ASPP_Block, self).__init__(**kwargs)
        self.filters=filters
        self.rate=rate
        self.name_prefix=name_prefix

        # First path
        self.first=tf.keras.layers.Conv2D(filters,(3,3),padding='same')
        #Second path
        self.second=tf.keras.layers.Conv2D(filters,(3,3),dilation_rate=(4*rate,4*rate),padding='same')
        #Third path
        self.third=tf.keras.layers.Conv2D(filters,(3,3),dilation_rate=(8*rate,8*rate),padding='same')
        #Fourth path
        self.fourth=tf.keras.layers.Conv2D(filters,(3,3),dilation_rate=(12*rate,12*rate),padding='same')
        #Add
        self.add=tf.keras.layers.Add()
        self.out=tf.keras.layers.Conv2D(filters,(1,1),padding='same',name=f"{name_prefix}_output")

    def call(self,inputs):
        x1=self.first(inputs)
        x2=self.second(inputs)
        x3=self.third(inputs)
        x4=self.fourth(inputs)
        x=self.add([x1,x2,x3,x4])
        x=self.out(x)
        return x
    
def build_custom_inceptionunet():
    inp=tf.keras.layers.Input(shape=(256,256,3))
    
    #encoder
    enc1=InceptionEncoder(16,32,name_prefix='encoder_1')(inp)
    enc2=InceptionEncoder(32,64,name_prefix='encoder_2')(enc1)
    enc3=InceptionEncoder(64,128,name_prefix='encoder_3')(enc2)
    enc4=InceptionEncoder(128,256,name_prefix='encoder_4')(enc3)

    #bridge
    bridge=ASPP_Block(256,name_prefix='bridge')(enc4)

    #decoder
    dec1=tf.keras.layers.Concatenate()([bridge,enc4])
    dec1=InceptionDecoder(128,128,name_prefix='decoder_1')(dec1)
    dec2=tf.keras.layers.UpSampling2D((2,2))(dec1)
    dec2=tf.keras.layers.Concatenate()([dec2,enc3])
    dec2=InceptionDecoder(64,64,name_prefix='decoder_2')(dec2)
    dec3=tf.keras.layers.UpSampling2D((2,2))(dec2)
    dec3=tf.keras.layers.Concatenate()([dec3,enc2])
    dec3=InceptionDecoder(32,32,name_prefix='decoder_3')(dec3)
    dec4=tf.keras.layers.UpSampling2D((2,2))(dec3)
    dec4=tf.keras.layers.Concatenate()([dec4,enc1])
    dec4=InceptionDecoder(16,16,name_prefix='decoder_4')(dec4)

    out=tf.keras.layers.UpSampling2D((2,2))(dec4)
    out=ASPP_Block(3,name_prefix='out_aspp',rate=1)(out)
    out=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(out)
    
    
    model=tf.keras.Model(inputs=inp,outputs=out)
    return model