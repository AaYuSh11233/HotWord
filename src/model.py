import tensorflow as tf
from tensorflow import keras
from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, L2_LAMBDA

class AdvancedWakeWordModel(keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.bn2 = keras.layers.BatchNormalization()
        self.pool1 = keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = keras.layers.Dropout(DROPOUT_RATE)
        
        self.conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.bn4 = keras.layers.BatchNormalization()
        self.pool2 = keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = keras.layers.Dropout(DROPOUT_RATE)
        
        self.reshape = keras.layers.Reshape((-1, 64))
        self.gru1 = keras.layers.GRU(64, return_sequences=True,
                                     kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.gru2 = keras.layers.GRU(64, kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.dropout3 = keras.layers.Dropout(DROPOUT_RATE)
        
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))
        self.bn5 = keras.layers.BatchNormalization()
        self.dropout4 = keras.layers.Dropout(DROPOUT_RATE)
        self.dense2 = keras.layers.Dense(NUM_CLASSES, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.reshape(x)
        x = self.gru1(x)
        x = self.gru2(x)
        x = self.dropout3(x, training=training)
        
        x = self.dense1(x)
        x = self.bn5(x, training=training)
        x = self.dropout4(x, training=training)
        return self.dense2(x)

def create_model():
    inputs = keras.Input(shape=INPUT_SHAPE)
    model = AdvancedWakeWordModel()
    outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

