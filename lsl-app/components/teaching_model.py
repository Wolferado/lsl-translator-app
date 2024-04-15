from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from keras.saving import save_model
from keras.utils import plot_model
from symbol_library import symbols_lib


import joblib
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class ModelCreator():
    def __init__(self):
        self.directory = os.path.join(os.curdir, "sign_data_wall")
        self.teached_symbols = symbols_lib
        self.sign_folders = os.listdir(self.directory) # All folders
        #self.sign_folders = ['a', 'b', 'c'] # For testing purposes
        self.max_frame_amount = 30
        self.face_points_amount = 366 # 1404 - whole face, 366 - limited face
        self.hand_points_amount = 63 # 63 - not traceable, 153 - traceable (6 points), 108 traceable (3 points)
        self.X_data = None
        self.Y_data = None
        self.test_size = 0.05
        self.model = None
        self.history = None
        self.main()

    def main(self):
        # Extract data from materials in the directory
        self.get_data_from_directory()

        # TEST SIZE: 0.05
        # self.test_size=0.05
        # self.lstm_model_compile()
        # self.save_lstm_model_and_graphs()

        # self.simplernn_model_compile()
        # self.save_simplernn_model_and_graphs()
        
        # self.random_forest_compile_and_save()

        # TEST SIZE: 0.10
        self.test_size = 0.10
        self.lstm_model_compile()
        self.save_lstm_model_and_graphs()

        # self.simplernn_model_compile()
        # self.save_simplernn_model_and_graphs()
        
        # self.random_forest_compile_and_save()

        # TEST SIZE: 0.15
        # self.test_size = 0.15
        # self.lstm_model_compile()
        # self.save_lstm_model_and_graphs()

        # self.simplernn_model_compile()
        # self.save_simplernn_model_and_graphs()
        
        # self.random_forest_compile_and_save()

        # # TEST SIZE: 0.40
        # self.test_size = 0.40
        # self.lstm_model_compile()
        # self.save_lstm_model_and_graphs()

        # self.simplernn_model_compile()
        # self.save_simplernn_model_and_graphs()
        
        # self.random_forest_compile_and_save()

    def get_data_from_directory(self):
        labels_enum = {label:num for num, label in enumerate(self.sign_folders)}

        print("Enumerated labels: ", labels_enum)

        data_collection = []
        labels = []

        last_time = datetime.now()

        for data_folder in self.sign_folders: # For each data folder in array of folder names
            print(datetime.now(), ": Processing ", data_folder, " (previous folder processed in {})".format(datetime.now() - last_time)) # For debugging
            last_time = datetime.now()
            for sub_folder in os.listdir(os.path.join(self.directory, data_folder)): # For each subfolder in the folder of the sign_folders
                data = [] # Array to store the values
                invalid_data = False

                for frame in range(0, self.max_frame_amount): # From 1st until maximum allowed frame
                    current_file = os.path.join(self.directory, data_folder, sub_folder, "{}.npy".format(frame)) # From each .npy file

                    if (os.path.isfile(current_file) and os.path.getsize(current_file) > 0):
                        pass
                        # res = np.load(current_file) # Get info from that .npy file
                        # data.append(res) # And append it to the window
                    else: 
                        invalid_data = True

                if invalid_data == False: # If there are data in the array
                    # data_collection.append(data) # Append data to sequence of defined letter or word
                    labels.append(labels_enum[data_folder]) # Append name of the folder along with its index

        # joblib.dump(data_collection, 'sign_data_wall_no_tracing.joblib') # Save data to stop endless pointless extraction

        print(datetime.now(), ": Processing finished", "(last folder processed in {})".format(datetime.now() - last_time)) # For debugging

        sign_data = joblib.load('sign_data_wall_no_tracing.joblib')

        self.X_data = np.array(sign_data) # 3D array of all data from frames. [file[frames[data]]]
        self.Y_data = to_categorical(labels) # 2D array for categories in both 

    def lstm_model_compile(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=self.test_size)

        #print(self.X_data.shape)
        #print(self.Y_data.shape)

        self.model = Sequential([
            Input(shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))), # 1530 - non-traceable, 1710 - traceable
            LSTM(128, return_sequences=True, activation='sigmoid'),
            Dropout(0.3),
            LSTM(128, return_sequences=False, activation='sigmoid'),
            Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
        ])

        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.005,
            patience=10,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=75,
        )

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train, epochs=200, validation_data=(self.x_test, self.y_test), callbacks=[early_stop])

        self.model.summary()

    def simplernn_model_compile(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=self.test_size)

        #print(self.X_data.shape)
        #print(self.Y_data.shape)
        
        self.model = Sequential([
            Input(shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))),
            SimpleRNN(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            SimpleRNN(64, return_sequences=False, activation='relu'),
            Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
        ])

        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.005,
            patience=20,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=80,
        )

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train, epochs=200, validation_data=(self.x_test, self.y_test), callbacks=[early_stop])

        self.model.summary()

    def save_lstm_model_and_graphs(self):
        # Accuracy graph
        plt.plot(self.history.history['categorical_accuracy'], color='#0277bd')
        plt.plot(self.history.history['val_categorical_accuracy'], linestyle='dashed', color='#f77500')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.title('LSTM model categorical accuracy overtime (test_size - {})'.format(self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('categorical_accuracy')
        plt.savefig("LSTM - Accuracy (test_size - {}).png".format(self.test_size))
        plt.clf()

        # Loss graph
        plt.plot(self.history.history['loss'], color='#0277bd')
        plt.plot(self.history.history['val_loss'], linestyle='dashed', color='#f77500')
        plt.legend(['loss', 'val_loss'])
        plt.title('LSTM model loss overtime (test_size - {})'.format(self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("LSTM - Loss (test_size - {}).png".format(self.test_size))
        plt.clf()

        ypred = self.model.predict(self.x_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        ypred = np.argmax(ypred, axis=1).tolist()
        conf_matrix = confusion_matrix(ypred, ytrue)

        # Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(15, 15))
        conf_matrix_figure = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.teached_symbols)
        conf_matrix_figure.plot(cmap=plt.cm.YlOrRd, ax=ax)
        plt.title('LSTM model confusion matrix (test_size - {})'.format(self.test_size))
        plt.savefig("LSTM - Confusion Matrix (test_size - {}).png".format(self.test_size))
        plt.clf()

        plot_model(model=self.model, to_file='lstm_model_structure.png', show_shapes=True)
        save_model(self.model, 'lstm_model_({}).keras'.format(self.test_size))

        fig, ax = plt.subplots(figsize=(8, 6))

    def save_simplernn_model_and_graphs(self):
        # Accuracy graph
        plt.plot(self.history.history['categorical_accuracy'], color='#0277bd')
        plt.plot(self.history.history['val_categorical_accuracy'], linestyle='dashed', color='#f77500')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.title('RNN model categorical accuracy overtime (test_size - {})'.format(self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('categorical_accuracy')
        plt.savefig("RNN - Accuracy (test_size - {}).png".format(self.test_size))
        plt.clf()

        # Loss graph
        plt.plot(self.history.history['loss'], color='#0277bd')
        plt.plot(self.history.history['val_loss'], linestyle='dashed', color='#f77500')
        plt.legend(['loss', 'val_loss'])
        plt.title('RNN model loss overtime (test_size:{})'.format(self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("RNN - Loss (test_size - {}).png".format(self.test_size))
        plt.clf()

        ypred = self.model.predict(self.x_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        ypred = np.argmax(ypred, axis=1).tolist()
        conf_matrix = confusion_matrix(ypred, ytrue)

        # Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(15, 15))
        conf_matrix_figure = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.teached_symbols)
        conf_matrix_figure.plot(cmap=plt.cm.YlOrRd, ax=ax)
        plt.title('RNN model confusion matrix (test_size - {})'.format(self.test_size))
        plt.savefig("RNN - Confusion Matrix (test_size - {}).png".format(self.test_size))
        plt.clf()

        plot_model(model=self.model, to_file='simplernn_model_structure.png', show_shapes=True)
        save_model(self.model, 'rnn_model_({}).keras'.format(self.test_size))

        fig, ax = plt.subplots(figsize=(8, 6))

    def random_forest_compile_and_save(self):
        # Reshaping 3D array, so it will be 2D array with first column being the same with Y_data.
        X_data = self.X_data
        X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_data, self.Y_data, test_size=self.test_size)

        print(self.X_data.shape) # (2010, 30, 672)
        print(X_data.shape) # (2010, 20160)
        print(self.Y_data.shape) # (2010, 34)

        number_of_trees = 30
        self.model = RandomForestClassifier(n_estimators=number_of_trees, criterion="gini", warm_start=True)

        self.history = self.model.fit(self.x_train, self.y_train)

        y_predict = self.model.predict(self.x_test)

        print("Accuracy score: ", accuracy_score(self.y_test, y_predict))
        print("Log loss: ", log_loss(self.y_test, y_predict))

        # Create the confusion matrix
        ypred = self.model.predict(self.x_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        ypred = np.argmax(ypred, axis=1).tolist()
        conf_matrix = confusion_matrix(ypred, ytrue)

        # Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(15, 15))
        conf_matrix_figure = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.teached_symbols)
        conf_matrix_figure.plot(cmap=plt.cm.YlOrRd, ax=ax)
        plt.title('Random Forest model confusion matrix (test_size - {})'.format(self.test_size))
        plt.savefig("Random Forest - Confusion Matrix (test_size - {}).png".format(self.test_size))
        plt.clf()

        file_name = "random_forest_model_(trees - {}, test_size - {}, acc - {}, loss - {}).joblib".format(number_of_trees, self.test_size, round(accuracy_score(self.y_test, y_predict), 2), round(log_loss(self.y_test, y_predict), 2))
        joblib.dump(self.model, file_name)

        fig, ax = plt.subplots(figsize=(8, 6))

    def count_files(self, directory) -> int:
        total_file_amount = 0

        for file in os.scandir(directory):
            if file.is_file():
                total_file_amount += 1

        return total_file_amount

if __name__ == "__main__":
    model_creator = ModelCreator()