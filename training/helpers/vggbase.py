from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.constraints import maxnorm

def VGG_BASE(include_top=True, input_shape=(32,32,3), classes=10):
    img_input = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(axis=-1, name='block1_bn1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(axis=-1, name='block1_bn2')(x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)
    x = Dropout(0.5, name='block1_drop')(x)


    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization(axis=-1, name='block2_bn1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(axis=-1, name='block2_bn2')(x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)
    x = Dropout(0.5, name='block2_drop')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(axis=-1, name='block3_bn1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(axis=-1, name='block3_bn2')(x)
    x = MaxPooling2D((2, 2), name='block3_pool')(x)
    x = Dropout(0.5, name='block3_drop')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='fc1')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(512, activation='relu', kernel_constraint=maxnorm(3), name='fc2')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='vgg')
    return model