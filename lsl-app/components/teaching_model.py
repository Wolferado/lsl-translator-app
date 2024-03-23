from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

import os
import numpy as np

directory = r"C:\Users\akare\Documents\RTU\Bakalaura darbs\sign-translation-prototype\sign_data"
folder_names = os.listdir(directory)
labels_enum = {label:num for num, label in enumerate(folder_names)}

sequences = []
labels = []

for folder in folder_names:
    for sequence in os.listdir(os.path.join(directory, folder)):
        window = []
        for frame_number in range(0, 100): # From 1st until 100th frame
            current_file = os.path.join(directory, folder, sequence, "{}.npy".format(frame_number)) # From each .npy file
            if os.path.isfile(current_file) == True: # If file is present
                if(os.path.getsize(current_file) > 0): # If file contains information
                    res = np.load(os.path.join(current_file)) # Get info from that .npy file
                    window.append(res) # And append it to the window
                else: # Otherwise, if file doesn't contain information, assign only zeros to it
                    res = np.concatenate([np.zeros(1404), np.zeros(63), np.zeros(63)])
                    window.append(res)
            else: # Otherwise, if there isn't file, assign only zeros to it
                res = np.concatenate([np.zeros(1404), np.zeros(63), np.zeros(63)])
                window.append(res)
        
        sequences.append(window)
        labels.append(labels_enum[folder])

print(np.array(sequences).shape)

X = np.array(sequences)
Y = to_categorical(labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
print(X.shape)
print(Y.shape)

log_dir = os.path.join(r"C:\Users\akare\Documents\RTU\Bakalaura darbs\sign-translation-prototype\logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(100,1530)))
model.add(LSTM(128, return_sequences=True, activation='sigmoid'))
model.add(LSTM(64, return_sequences=False, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(np.array(folder_names).shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, epochs=500, callbacks=[tb_callback])

model.summary()

model.save('signs_model.h5')

model.load_weights('signs_model.h5')