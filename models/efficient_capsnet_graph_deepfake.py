# efficient_capsnet_graph_deepfake.py

import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length, Mask


def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture for deepfake detection.

    Parameters
    ----------
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    print("deepfake Input shape:", inputs.shape)  # 添加这行代码

    # 定义卷积层
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # 定义主胶囊层
    x = PrimaryCaps(128, 9, 16, 8)(x)

    # 定义数字胶囊层
    digit_caps = FCCaps(2, 16)(x)  # 对于deepfake检测,有2个类别(真实和伪造)

    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name='Efficient_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.

    Parameters
    ----------
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16 * 2)  # 对于deepfake检测,有2个类别

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    # x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    # x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    # x = tf.keras.layers.Dense(3888, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer for deepfake detection.

    Parameters
    ----------
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    print("build_graph deepfake Input shape:", inputs.shape)  # 添加这行代码

    y_true = tf.keras.layers.Input(shape=(2,))  # 对于deepfake检测,有2个类别
    noise = tf.keras.layers.Input(shape=(2, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")

    digit_caps, digit_caps_len = efficient_capsnet(inputs)

    # 新增一行,获取第二层全连接层的输出维度
    # prev_output_dim = digit_caps.shape[-1]

    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise])

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])

    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    # 修改第三层全连接层的定义
    # x_gen_train = generator(masked_by_y[:, :prev_output_dim])
    # x_gen_eval = generator(masked[:, :prev_output_dim])
    # x_gen_play = generator(masked_noised_y[:, :prev_output_dim])


    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train],
                                     name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficinet_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play],
                                     name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')