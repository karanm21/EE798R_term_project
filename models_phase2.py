# import os

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import Model

# import hyperparameters

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# def Light_SERNet_V1(output_class,
#                     input_duration,
#                     input_type="mfcc"):

    
#     number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
#     if input_type == "mfcc":
#         number_of_feature = hyperparameters.N_MFCC
#         number_of_channel = 1
#     elif input_type == "spectrogram":
#         number_of_feature = hyperparameters.NUM_SPECTROGRAM_BINS
#         number_of_channel = 1
#     elif input_type == "mel_spectrogram":
#         number_of_feature = hyperparameters.NUM_MEL_BINS
#         number_of_channel = 1
#     else:
#         raise ValueError('input_type not valid!')


#     body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

#     path1 = layers.Conv2D(32, (11,1), padding="same", strides=(1,1))(body_input)
#     path2 = layers.Conv2D(32, (1, 9), padding="same", strides=(1,1))(body_input)
#     path3 = layers.Conv2D(32, (3, 3), padding="same", strides=(1,1))(body_input)

#     path1 = layers.BatchNormalization()(path1)
#     path2 = layers.BatchNormalization()(path2)
#     path3 = layers.BatchNormalization()(path3)

#     path1 = layers.ReLU()(path1)
#     path2 = layers.ReLU()(path2)
#     path3 = layers.ReLU()(path3)

#     path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)
#     path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)
#     path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)


#     feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

#     x = layers.Conv2D(64, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(96, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2,2), padding="same")(x)


#     x = layers.Conv2D(128, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2,1) , padding="same")(x)

#     x = layers.Conv2D(160, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2,1) , padding="same")(x)

#     x = layers.Conv2D(320, (1,1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.GlobalAveragePooling2D()(x)


#     x = layers.Dropout(hyperparameters.DROPOUT)(x)


#     body_output = layers.Dense(output_class, activation="softmax")(x)
#     body_model = Model(inputs=body_input, outputs=body_output)

#     return body_model



# class MFCCExtractor(tf.keras.layers.Layer):
#     def __init__(self,
#                     NUM_MEL_BINS,
#                     SAMPLE_RATE,
#                     LOWER_EDGE_HERTZ,
#                     UPPER_EDGE_HERTZ,
#                     FRAME_LENGTH,
#                     FRAME_STEP,
#                     N_MFCC,
#                     **kwargs):
#         super(MFCCExtractor, self).__init__(**kwargs)

#         self.NUM_MEL_BINS = NUM_MEL_BINS
#         self.SAMPLE_RATE = SAMPLE_RATE
#         self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
#         self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ

#         self.FRAME_LENGTH = FRAME_LENGTH
#         self.FRAME_STEP = FRAME_STEP

#         self.N_MFCC = N_MFCC


#     def get_mfcc(self, waveform, clip_value=10):
#         waveform = tf.cast(waveform, tf.float32)
#         spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform,
#                                                     window_size=self.FRAME_LENGTH,
#                                                     stride=self.FRAME_STEP,
#                                                     magnitude_squared=True,
#                                                     )


#         mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram,
#                                 sample_rate=hyperparameters.SAMPLE_RATE,
#                                 upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
#                                 lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
#                                 filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
#                                 dct_coefficient_count=hyperparameters.N_MFCC,
#                                 )

#         return tf.clip_by_value(mfcc, -clip_value, clip_value)


#     def call(self, inputs):
#         outputs = self.get_mfcc(inputs)

#         return tf.expand_dims(outputs, -1)


#     def get_config(self):
#         config = super(MFCCExtractor, self).get_config()
#         config.update({
#             "NUM_MEL_BINS": self.NUM_MEL_BINS,
#             "SAMPLE_RATE": self.SAMPLE_RATE,
#             "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
#             "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
#             "FRAME_LENGTH": self.FRAME_LENGTH,
#             "FRAME_STEP": self.FRAME_STEP,
#             "N_MFCC": self.N_MFCC,
#         })
#         return config



#attention

# import os
# import tensorflow as tf
# from tensorflow.keras import layers, regularizers, Model
# from tensorflow.keras import backend as K
# import hyperparameters

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def Light_SERNet_V1(output_class, input_duration, input_type="mfcc"):
#     number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
#     if input_type == "mfcc":
#         number_of_feature = hyperparameters.N_MFCC
#         number_of_channel = 1
#     elif input_type == "spectrogram":
#         number_of_feature = hyperparameters.NUM_SPECTROGRAM_BINS
#         number_of_channel = 1
#     elif input_type == "mel_spectrogram":
#         number_of_feature = hyperparameters.NUM_MEL_BINS
#         number_of_channel = 1
#     else:
#         raise ValueError('input_type not valid!')

#     body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

#     path1 = layers.Conv2D(32, (11, 1), padding="same", strides=(1, 1))(body_input)
#     path2 = layers.Conv2D(32, (1, 9), padding="same", strides=(1, 1))(body_input)
#     path3 = layers.Conv2D(32, (3, 3), padding="same", strides=(1, 1))(body_input)

#     path1 = layers.BatchNormalization()(path1)
#     path2 = layers.BatchNormalization()(path2)
#     path3 = layers.BatchNormalization()(path3)

#     path1 = layers.ReLU()(path1)
#     path2 = layers.ReLU()(path2)
#     path3 = layers.ReLU()(path3)

#     path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)
#     path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)
#     path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)

#     feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

#     # Main convolutional blocks
#     x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(96, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     # Flatten dimensions for Attention layer
#     original_shape = K.int_shape(x)
#     flattened_dim = original_shape[1] * original_shape[2]
#     x = layers.Reshape((flattened_dim, original_shape[-1]))(x)

#     # Attention layer
#     attention_output = layers.Attention()([x, x])
#     x = layers.Add()([x, attention_output])  # Add residual connection

#     # Reshape back to the closest shape we had before flattening
#     reshaped_height = original_shape[1]
#     reshaped_width = original_shape[2]
#     x = layers.Reshape((reshaped_height, reshaped_width, 128))(x)

#     x = layers.Conv2D(160, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

#     x = layers.Conv2D(320, (1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dropout(hyperparameters.DROPOUT)(x)
#     body_output = layers.Dense(output_class, activation="softmax")(x)

#     return Model(inputs=body_input, outputs=body_output)

# class MFCCExtractor(tf.keras.layers.Layer):
#     def __init__(self, NUM_MEL_BINS, SAMPLE_RATE, LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ, FRAME_LENGTH, FRAME_STEP, N_MFCC, **kwargs):
#         super(MFCCExtractor, self).__init__(**kwargs)
#         self.NUM_MEL_BINS = NUM_MEL_BINS
#         self.SAMPLE_RATE = SAMPLE_RATE
#         self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
#         self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ
#         self.FRAME_LENGTH = FRAME_LENGTH
#         self.FRAME_STEP = FRAME_STEP
#         self.N_MFCC = N_MFCC

#     def get_mfcc(self, waveform, clip_value=10):
#         waveform = tf.cast(waveform, tf.float32)
#         spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform, window_size=self.FRAME_LENGTH, stride=self.FRAME_STEP, magnitude_squared=True)
#         mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram, sample_rate=hyperparameters.SAMPLE_RATE,
#                                upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
#                                lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
#                                filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
#                                dct_coefficient_count=hyperparameters.N_MFCC)
#         return tf.clip_by_value(mfcc, -clip_value, clip_value)

#     def call(self, inputs):
#         outputs = self.get_mfcc(inputs)
#         return tf.expand_dims(outputs, -1)

#     def get_config(self):
#         config = super(MFCCExtractor, self).get_config()
#         config.update({
#             "NUM_MEL_BINS": self.NUM_MEL_BINS,
#             "SAMPLE_RATE": self.SAMPLE_RATE,
#             "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
#             "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
#             "FRAME_LENGTH": self.FRAME_LENGTH,
#             "FRAME_STEP": self.FRAME_STEP,
#             "N_MFCC": self.N_MFCC,
#         })
#         return config



# Depthwise Separable Convolutions

# import os

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import Model

# import hyperparameters

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# def Light_SERNet_V1(output_class,
#                     input_duration,
#                     input_type="mfcc"):

#     number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
#     if input_type == "mfcc":
#         number_of_feature = hyperparameters.N_MFCC
#         number_of_channel = 1
#     elif input_type == "spectrogram":
#         number_of_feature = hyperparameters.NUM_SPECTROGRAM_BINS
#         number_of_channel = 1
#     elif input_type == "mel_spectrogram":
#         number_of_feature = hyperparameters.NUM_MEL_BINS
#         number_of_channel = 1
#     else:
#         raise ValueError('input_type not valid!')

#     body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

#     # Using Depthwise Separable Convolutions in initial paths
#     path1 = layers.SeparableConv2D(32, (11, 1), padding="same", strides=(1, 1))(body_input)
#     path2 = layers.SeparableConv2D(32, (1, 9), padding="same", strides=(1, 1))(body_input)
#     path3 = layers.SeparableConv2D(32, (3, 3), padding="same", strides=(1, 1))(body_input)

#     path1 = layers.BatchNormalization()(path1)
#     path2 = layers.BatchNormalization()(path2)
#     path3 = layers.BatchNormalization()(path3)

#     path1 = layers.ReLU()(path1)
#     path2 = layers.ReLU()(path2)
#     path3 = layers.ReLU()(path3)

#     path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)
#     path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)
#     path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)

#     feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

#     x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(96, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

#     x = layers.Conv2D(160, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

#     x = layers.Conv2D(320, (1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dropout(hyperparameters.DROPOUT)(x)

#     body_output = layers.Dense(output_class, activation="softmax")(x)
#     body_model = Model(inputs=body_input, outputs=body_output)

#     return body_model


# class MFCCExtractor(tf.keras.layers.Layer):
#     def __init__(self,
#                  NUM_MEL_BINS,
#                  SAMPLE_RATE,
#                  LOWER_EDGE_HERTZ,
#                  UPPER_EDGE_HERTZ,
#                  FRAME_LENGTH,
#                  FRAME_STEP,
#                  N_MFCC,
#                  **kwargs):
#         super(MFCCExtractor, self).__init__(**kwargs)

#         self.NUM_MEL_BINS = NUM_MEL_BINS
#         self.SAMPLE_RATE = SAMPLE_RATE
#         self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
#         self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ

#         self.FRAME_LENGTH = FRAME_LENGTH
#         self.FRAME_STEP = FRAME_STEP

#         self.N_MFCC = N_MFCC

#     def get_mfcc(self, waveform, clip_value=10):
#         waveform = tf.cast(waveform, tf.float32)
#         spectrogram = tf.raw_ops.AudioSpectrogram(
#             input=waveform,
#             window_size=self.FRAME_LENGTH,
#             stride=self.FRAME_STEP,
#             magnitude_squared=True,
#         )

#         mfcc = tf.raw_ops.Mfcc(
#             spectrogram=spectrogram,
#             sample_rate=self.SAMPLE_RATE,
#             upper_frequency_limit=self.UPPER_EDGE_HERTZ,
#             lower_frequency_limit=self.LOWER_EDGE_HERTZ,
#             filterbank_channel_count=self.NUM_MEL_BINS,
#             dct_coefficient_count=self.N_MFCC,
#         )

#         return tf.clip_by_value(mfcc, -clip_value, clip_value)

#     def call(self, inputs):
#         outputs = self.get_mfcc(inputs)
#         return tf.expand_dims(outputs, -1)

#     def get_config(self):
#         config = super(MFCCExtractor, self).get_config()
#         config.update({
#             "NUM_MEL_BINS": self.NUM_MEL_BINS,
#             "SAMPLE_RATE": self.SAMPLE_RATE,
#             "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
#             "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
#             "FRAME_LENGTH": self.FRAME_LENGTH,
#             "FRAME_STEP": self.FRAME_STEP,
#             "N_MFCC": self.N_MFCC,
#         })
#         return config


#hybrid

# import os
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from tensorflow.keras.models import Model

# import hyperparameters

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# class ScaledDotProductAttention(layers.Layer):
#     def __init__(self, d_model, **kwargs):
#         super(ScaledDotProductAttention, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.sqrt_d = tf.math.sqrt(tf.cast(d_model, tf.float32))

#     def call(self, query, key, value):
#         scores = tf.matmul(query, key, transpose_b=True) / self.sqrt_d
#         attention_weights = tf.nn.softmax(scores, axis=-1)
#         return tf.matmul(attention_weights, value), attention_weights

#     def get_config(self):
#         config = super().get_config()
#         config.update({"d_model": self.d_model})
#         return config


# def Light_SERNet_V1(output_class, input_duration, input_type="mfcc"):
#     number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
#     number_of_feature = {
#         "mfcc": hyperparameters.N_MFCC,
#         "spectrogram": hyperparameters.NUM_SPECTROGRAM_BINS,
#         "mel_spectrogram": hyperparameters.NUM_MEL_BINS
#     }.get(input_type, 1)

#     body_input = layers.Input(shape=(number_of_frame, number_of_feature, 1))

#     # Initial convolutional paths with hybrid parallel attention
#     path1 = layers.Conv2D(32, (11, 1), padding="same", strides=(1, 1))(body_input)
#     path2 = layers.Conv2D(32, (1, 9), padding="same", strides=(1, 1))(body_input)
#     path3 = layers.Conv2D(32, (3, 3), padding="same", strides=(1, 1))(body_input)

#     path1 = layers.BatchNormalization()(path1)
#     path2 = layers.BatchNormalization()(path2)
#     path3 = layers.BatchNormalization()(path3)

#     path1 = layers.ReLU()(path1)
#     path2 = layers.ReLU()(path2)
#     path3 = layers.ReLU()(path3)

#     # Parallel attention to add hybrid characteristics
#     query = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(body_input)
#     key = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(body_input)
#     value = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(body_input)
#     attention_output, _ = ScaledDotProductAttention(d_model=32)(query, key, value)

#     # Concatenate paths and attention
#     feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3, attention_output])

#     x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     # Mid-layer attention
#     query = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(x)
#     key = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(x)
#     value = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(x)
#     attention_output, _ = ScaledDotProductAttention(d_model=64)(query, key, value)
#     x = tf.keras.layers.Add()([x, attention_output])

#     x = layers.Conv2D(96, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     # Final layers
#     x = layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dropout(hyperparameters.DROPOUT)(x)
#     body_output = layers.Dense(output_class, activation="softmax")(x)
#     body_model = Model(inputs=body_input, outputs=body_output)

#     return body_model


# class MFCCExtractor(tf.keras.layers.Layer):
#     def __init__(self,
#                     NUM_MEL_BINS,
#                     SAMPLE_RATE,
#                     LOWER_EDGE_HERTZ,
#                     UPPER_EDGE_HERTZ,
#                     FRAME_LENGTH,
#                     FRAME_STEP,
#                     N_MFCC,
#                     **kwargs):
#         super(MFCCExtractor, self).__init__(**kwargs)

#         self.NUM_MEL_BINS = NUM_MEL_BINS
#         self.SAMPLE_RATE = SAMPLE_RATE
#         self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
#         self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ

#         self.FRAME_LENGTH = FRAME_LENGTH
#         self.FRAME_STEP = FRAME_STEP

#         self.N_MFCC = N_MFCC


#     def get_mfcc(self, waveform, clip_value=10):
#         waveform = tf.cast(waveform, tf.float32)
#         spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform,
#                                                     window_size=self.FRAME_LENGTH,
#                                                     stride=self.FRAME_STEP,
#                                                     magnitude_squared=True,
#                                                     )


#         mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram,
#                                 sample_rate=hyperparameters.SAMPLE_RATE,
#                                 upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
#                                 lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
#                                 filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
#                                 dct_coefficient_count=hyperparameters.N_MFCC,
#                                 )

#         return tf.clip_by_value(mfcc, -clip_value, clip_value)


#     def call(self, inputs):
#         outputs = self.get_mfcc(inputs)

#         return tf.expand_dims(outputs, -1)


#     def get_config(self):
#         config = super(MFCCExtractor, self).get_config()
#         config.update({
#             "NUM_MEL_BINS": self.NUM_MEL_BINS,
#             "SAMPLE_RATE": self.SAMPLE_RATE,
#             "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
#             "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
#             "FRAME_LENGTH": self.FRAME_LENGTH,
#             "FRAME_STEP": self.FRAME_STEP,
#             "N_MFCC": self.N_MFCC,
#         })
#         return config



# skip connections

# import os
# import tensorflow as tf
# from tensorflow.keras import layers, regularizers
# from tensorflow.keras.models import Model
# import hyperparameters

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def Light_SERNet_V1(output_class, input_duration, input_type="mfcc"):
#     number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
#     if input_type == "mfcc":
#         number_of_feature = hyperparameters.N_MFCC
#         number_of_channel = 1
#     elif input_type == "spectrogram":
#         number_of_feature = hyperparameters.NUM_SPECTROGRAM_BINS
#         number_of_channel = 1
#     elif input_type == "mel_spectrogram":
#         number_of_feature = hyperparameters.NUM_MEL_BINS
#         number_of_channel = 1
#     else:
#         raise ValueError('input_type not valid!')

#     body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

#     path1 = layers.Conv2D(32, (11, 1), padding="same", strides=(1, 1))(body_input)
#     path1 = layers.BatchNormalization()(path1)
#     path1 = layers.ReLU()(path1)
#     path1_skip = layers.Conv2D(32, (11, 1), padding="same", strides=(1, 1))(path1)
#     path1 = layers.Add()([path1, path1_skip])
#     path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)

#     path2 = layers.Conv2D(32, (1, 9), padding="same", strides=(1, 1))(body_input)
#     path2 = layers.BatchNormalization()(path2)
#     path2 = layers.ReLU()(path2)
#     path2_skip = layers.Conv2D(32, (1, 9), padding="same", strides=(1, 1))(path2)
#     path2 = layers.Add()([path2, path2_skip])
#     path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)

#     path3 = layers.Conv2D(32, (3, 3), padding="same", strides=(1, 1))(body_input)
#     path3 = layers.BatchNormalization()(path3)
#     path3 = layers.ReLU()(path3)
#     path3_skip = layers.Conv2D(32, (3, 3), padding="same", strides=(1, 1))(path3)
#     path3 = layers.Add()([path3, path3_skip])
#     path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)

#     feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

#     x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x_skip = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
#     x = layers.Add()([x, x_skip])
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(96, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x_skip = layers.Conv2D(96, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.Add()([x, x_skip])
#     x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

#     x = layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x_skip = layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.Add()([x, x_skip])
#     x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

#     x = layers.Conv2D(160, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x_skip = layers.Conv2D(160, (3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.Add()([x, x_skip])
#     x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

#     x = layers.Conv2D(320, (1, 1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dropout(hyperparameters.DROPOUT)(x)

#     body_output = layers.Dense(output_class, activation="softmax")(x)
#     body_model = Model(inputs=body_input, outputs=body_output)

#     return body_model


# class MFCCExtractor(tf.keras.layers.Layer):
#     def __init__(self,
#                  NUM_MEL_BINS,
#                  SAMPLE_RATE,
#                  LOWER_EDGE_HERTZ,
#                  UPPER_EDGE_HERTZ,
#                  FRAME_LENGTH,
#                  FRAME_STEP,
#                  N_MFCC,
#                  **kwargs):
#         super(MFCCExtractor, self).__init__(**kwargs)
#         self.NUM_MEL_BINS = NUM_MEL_BINS
#         self.SAMPLE_RATE = SAMPLE_RATE
#         self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
#         self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ
#         self.FRAME_LENGTH = FRAME_LENGTH
#         self.FRAME_STEP = FRAME_STEP
#         self.N_MFCC = N_MFCC

#     def get_mfcc(self, waveform, clip_value=10):
#         waveform = tf.cast(waveform, tf.float32)
#         spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform, window_size=self.FRAME_LENGTH, stride=self.FRAME_STEP, magnitude_squared=True)
#         mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram,
#                                sample_rate=hyperparameters.SAMPLE_RATE,
#                                upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
#                                lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
#                                filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
#                                dct_coefficient_count=hyperparameters.N_MFCC)
#         return tf.clip_by_value(mfcc, -clip_value, clip_value)

#     def call(self, inputs):
#         outputs = self.get_mfcc(inputs)
#         return tf.expand_dims(outputs, -1)

#     def get_config(self):
#         config = super(MFCCExtractor, self).get_config()
#         config.update({
#             "NUM_MEL_BINS": self.NUM_MEL_BINS,
#             "SAMPLE_RATE": self.SAMPLE_RATE,
#             "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
#             "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
#             "FRAME_LENGTH": self.FRAME_LENGTH,
#             "FRAME_STEP": self.FRAME_STEP,
#             "N_MFCC": self.N_MFCC,
#         })
#         return config




# attention new


import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import hyperparameters

# Channel Attention Layer
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=True)
    shared_layer_two = layers.Dense(channel, use_bias=True)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.Multiply()([input_feature, cbam_feature])

# Updated Spatial Attention Layer using Keras operations
def spatial_attention(input_feature):
    # Use Keras layers to replace tf.reduce_mean and tf.reduce_max
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    return layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)


def Light_SERNet_V1(output_class, input_duration, input_type="mfcc"):
    number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - 
                       hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
    if input_type == "mfcc":
        number_of_feature = hyperparameters.N_MFCC
        number_of_channel = 1
    else:
        raise ValueError('input_type not valid!')

    body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

    # Path 1 - Temporal Path with Channel Attention
    path1 = layers.Conv2D(32, (11,1), padding="same", strides=(1,1))(body_input)
    path1 = layers.BatchNormalization()(path1)
    path1 = layers.ReLU()(path1)
    path1 = channel_attention(path1)  # Adding Channel Attention to Path 1
    path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)

    # Path 2 - Spectral Path
    path2 = layers.Conv2D(32, (1,9), padding="same", strides=(1,1))(body_input)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.ReLU()(path2)
    path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)

    # Path 3 - Spectral-Temporal Path with Spatial Attention
    path3 = layers.Conv2D(32, (3,3), padding="same", strides=(1,1))(body_input)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.ReLU()(path3)
    path3 = spatial_attention(path3)  # Adding Spatial Attention to Path 3
    path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)

    # Concatenate paths
    feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

    # Body Part II - LFLBs with Global Channel Attention in the last block
    x = layers.Conv2D(64, (3,3), strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

    x = layers.Conv2D(96, (3,3), strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

    x = layers.Conv2D(128, (3,3), strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

    # Adding Global Channel Attention in the final LFLB
    x = channel_attention(x)

    x = layers.Conv2D(160, (3,3), strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 1), padding="same")(x)

    x = layers.Conv2D(320, (1,1), strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Head - Classification
    x = layers.Dropout(hyperparameters.DROPOUT)(x)
    body_output = layers.Dense(output_class, activation="softmax")(x)
    body_model = Model(inputs=body_input, outputs=body_output)

    return body_model



class MFCCExtractor(tf.keras.layers.Layer):
    def __init__(self,
                    NUM_MEL_BINS,
                    SAMPLE_RATE,
                    LOWER_EDGE_HERTZ,
                    UPPER_EDGE_HERTZ,
                    FRAME_LENGTH,
                    FRAME_STEP,
                    N_MFCC,
                    **kwargs):
        super(MFCCExtractor, self).__init__(**kwargs)

        self.NUM_MEL_BINS = NUM_MEL_BINS
        self.SAMPLE_RATE = SAMPLE_RATE
        self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
        self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ

        self.FRAME_LENGTH = FRAME_LENGTH
        self.FRAME_STEP = FRAME_STEP

        self.N_MFCC = N_MFCC


    def get_mfcc(self, waveform, clip_value=10):
        waveform = tf.cast(waveform, tf.float32)
        spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform,
                                                    window_size=self.FRAME_LENGTH,
                                                    stride=self.FRAME_STEP,
                                                    magnitude_squared=True,
                                                    )


        mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=hyperparameters.SAMPLE_RATE,
                                upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
                                lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
                                filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
                                dct_coefficient_count=hyperparameters.N_MFCC,
                                )

        return tf.clip_by_value(mfcc, -clip_value, clip_value)


    def call(self, inputs):
        outputs = self.get_mfcc(inputs)

        return tf.expand_dims(outputs, -1)


    def get_config(self):
        config = super(MFCCExtractor, self).get_config()
        config.update({
            "NUM_MEL_BINS": self.NUM_MEL_BINS,
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
            "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
            "FRAME_LENGTH": self.FRAME_LENGTH,
            "FRAME_STEP": self.FRAME_STEP,
            "N_MFCC": self.N_MFCC,
        })
        return config