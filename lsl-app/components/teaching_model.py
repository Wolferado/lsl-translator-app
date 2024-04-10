import joblib
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from keras.saving import save_model
from keras.utils import plot_model

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class ModelCreator():
    def __init__(self):
        self.directory = os.path.join(os.curdir, "sign_data")
        self.sign_folders = os.listdir(self.directory) # All folders
        #self.sign_folders = ['a', 'b', 'c'] # Three letters
        self.max_frame_amount = 30
        self.face_points_amount = 366 # 1404 - whole face, 366 - limited face
        self.hand_points_amount = 153 # 63 - not traceable, 153 - traceable
        self.X_data = None
        self.Y_data = None
        self.model = None
        self.history = None
        self.main()

    def main(self):
        # Extract data from materials in the directory
        self.get_data_from_directory()

        # LSTM model
        self.create_and_compile_lstm_model()
        self.display_lstm_model_and_graphs()

        # RNN model
        self.create_and_compile_simplernn_model()
        self.display_simplernn_model_and_graphs()
        
        # Random Forest model
        self.create_and_compile_random_forest_model()

        input("Press any key to exit...")

    def get_data_from_directory(self):
        labels_enum = {label:num for num, label in enumerate(self.sign_folders)}

        print("Enumerated labels: ", labels_enum)

        data_collection = []
        labels = []

        last_time = datetime.now()

        for data_folder in self.sign_folders: # For each data folder in array of folder names
            print(datetime.now(), ", Processed ", data_folder, " ({})".format(datetime.now() - last_time)) # For debugging
            last_time = datetime.now()
            for sub_folder in os.listdir(os.path.join(self.directory, data_folder)): # For each subfolder in the folder of the sign_folders
                data = [] # Array to store the values

                for frame in range(0, self.max_frame_amount): # From 1st until maximum allowed frame
                    current_file = os.path.join(self.directory, data_folder, sub_folder, "{}.npy".format(frame)) # From each .npy file
                    res = np.load(current_file) # Get info from that .npy file
                    data.append(res) # And append it to the window

                data_collection.append(data) # Append data to sequence of defined letter or word
                labels.append(labels_enum[data_folder]) # Append name of the folder along with its index

        self.X_data = np.array(data_collection) # 3D array of all data from frames. [file[frames[data]]]
        self.Y_data = to_categorical(labels) # 2D array for categories in both 

    def create_and_compile_simplernn_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=0.05)

        print(self.X_data.shape)
        print(self.Y_data.shape)
        
        self.model = Sequential([
            SimpleRNN(64, return_sequences=True, activation='relu', input_shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))), # 1530 - non-traceable, 1710 - traceable
            SimpleRNN(128, return_sequences=True, activation='relu'),
            SimpleRNN(64, return_sequences=False, activation='relu'),
            Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
        ])

        self.model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train, epochs=100, validation_data=(self.x_test, self.y_test))

        self.model.summary()

    def create_and_compile_lstm_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=0.05)

        print(self.X_data.shape)
        print(self.Y_data.shape)
        
        self.model = Sequential([
            LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))), # 1530 - non-traceable, 1710 - traceable
            LSTM(32, return_sequences=False, activation='sigmoid'),
            Dense(32, activation='sigmoid'),
            Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train, epochs=150, validation_data=(self.x_test, self.y_test))

        self.model.summary()

        save_model(self.model, 'lstm_model.keras')

    def create_and_compile_random_forest_model(self):
        # Reshaping 3D array, so it will be 2D array with first column being the same with Y_data.
        self.X_data = self.X_data.reshape(self.X_data.shape[0], self.X_data.shape[1] * self.X_data.shape[2])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=0.05)

        print(self.X_data.shape)
        print(self.Y_data.shape)

        self.model = RandomForestClassifier(n_estimators=150)

        self.history = self.model.fit(self.x_train, self.y_train)

        y_predict = self.model.predict(self.x_test)

        print("Accuracy score: ", accuracy_score(self.y_test, y_predict))

        joblib.dump(self.model, "./random_forest_model.joblib")

    def display_lstm_model_and_graphs(self):
        plot_model(model=self.model, to_file='lstm_model_structure.png', show_shapes=True)

        print(self.history.history.keys())

        accuracy_figure = plt.figure(1)
        plt.plot(self.history.history['categorical_accuracy'], color='#0277bd')
        plt.plot(self.history.history['val_categorical_accuracy'], linestyle='dashed', color='#f77500')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.title('LSTM model categorical accuracy overtime')
        plt.xlabel('epoch')
        plt.ylabel('categorical_accuracy')
        accuracy_figure.show()

        loss_figure = plt.figure(2)
        plt.plot(self.history.history['loss'], color='#0277bd')
        plt.plot(self.history.history['val_loss'], linestyle='dashed', color='#f77500')
        plt.legend(['loss', 'val_loss'])
        plt.title('LSTM model loss overtime')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        loss_figure.show()

        plt.show()
    
    def display_simplernn_model_and_graphs(self):
        print(self.history.history.keys())

        accuracy_figure = plt.figure(1)
        plt.plot(self.history.history['categorical_accuracy'], color='#0277bd')
        plt.plot(self.history.history['val_categorical_accuracy'], linestyle='dashed', color='#f77500')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.title('RNN model categorical accuracy overtime')
        plt.xlabel('epoch')
        plt.ylabel('categorical_accuracy')
        accuracy_figure.show()

        loss_figure = plt.figure(2)
        plt.plot(self.history.history['loss'], color='#0277bd')
        plt.plot(self.history.history['val_loss'], linestyle='dashed', color='#f77500')
        plt.legend(['loss', 'val_loss'])
        plt.title('RNN model loss overtime')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        loss_figure.show()

        plt.show(block=False)

        while True:
            save_model_bool = input("Save model? (Y/n): ")
            if str.lower(save_model_bool[0]) == 'y':
                save_model(self.model, 'rnn_model.keras')
                break
            elif str.lower(save_model_bool[0]) == 'n':
                break

        while True:
            save_model_plot_bool = input("Save model plot? (Y/n): ")
            if str.lower(save_model_plot_bool[0]) == 'y':
                plot_model(model=self.model, to_file='simplernn_model_structure.png', show_shapes=True)
                break
            elif str.lower(save_model_bool[0]) == 'n':
                break

    def count_files(self, directory) -> int:
        total_file_amount = 0

        for file in os.scandir(directory):
            if file.is_file():
                total_file_amount += 1

        return total_file_amount

if __name__ == "__main__":
    model_creator = ModelCreator()