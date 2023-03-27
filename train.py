import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import data_generator

# Model parameters
max_features = 15000
max_len = 200
embedding_dim = 128
lstm_units = 64
hidden_units = 32
train_batch_size = 64
val_batch_size = 64

# Generator instances and steps per epoch calculations
train_steps_per_epoch = (os.path.getsize("data/train.ft.txt") // train_batch_size) + 1
val_steps_per_epoch = (os.path.getsize("data/test.ft.txt") // val_batch_size) + 1

train_generator = data_generator("data/train.ft.txt", train_batch_size, max_features, max_len)
val_generator = data_generator("data/test.ft.txt", val_batch_size, max_features, max_len)

# Model definition
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=10,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])
