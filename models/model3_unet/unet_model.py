# EXPERIMENTAL UNET MODEL
from keras.models import *
from keras.layers import *
input_shape=(image_height, image_width, 1)

inputs = Input(input_shape)

conv1f = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
conv1s = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv1f)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1s)

conv2f = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool1)
conv2s = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv2f)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2s)

conv3f = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool2)
conv3s = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv3f)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3s)

conv4f = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool3)
conv4s = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv4f)
drop4 = Dropout(0.5)(conv4s)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5f = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool4)
#conv5f = Activation("relu")(conv5f)
conv5s = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv5f)
drop5 = Dropout(0.5)(conv5s)

up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4s], axis=3)
conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up6)
#conv6 = Activation("relu")(conv6)
conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv6)
#conv6 = Activation("relu")(conv6)

up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3s], axis=3)
conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up7)
#conv7 = Activation("relu")(conv7)
conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv7)
#conv7 = Activation("relu")(conv7)

up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2s], axis=3)
conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up8)
#conv8 = Activation("relu")(conv8)
conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv8)
#conv8 = Activation("relu")(conv8)

up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1s], axis=3)
conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(up9)
#conv9 = Activation("relu")(conv9)
conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)
#conv9 = Activation("relu")(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

outputs = Conv2D(1, 1, activation='linear')(conv9)

model = Model(inputs=inputs, outputs=outputs)
