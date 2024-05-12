from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, GRU, Dropout, Input, GaussianNoise
from keras.metrics import Precision, Recall, F1Score, AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from keras.saving import save_model
from keras.utils import plot_model
from symbol_library import signs_lib

import joblib
import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class ModelCreator():
    def __init__(self):
        self.directory = os.path.join(os.curdir, "sign_data_full_collection")
        self.signs_to_learn = list(signs_lib.keys())
        self.sign_folders = os.listdir(self.directory) # All folders
        self.max_frame_amount = 30
        self.face_points_amount = 366 # 1404 - whole face, 366 - limited face
        self.hand_points_amount = 108 # 63 - not traceable, 153 - traceable (6 points), 108 traceable (3 points)
        self.X_data = None
        self.Y_data = None
        self.test_size = 0.35
        self.model = None
        self.history = None
        self.main()

    def main(self):
        # Extract data from materials in the directory
        self.get_data_from_directory()

        # Test size
        # self.test_size = 0.35

        # self.model_compile(model_name="RNN", value_to_monitor="val_loss", min_delta_value=0.1, mode_option="min", starting_epoch=110, optimizer_name="adam", num_of_epochs=150)
        # self.save_model_and_graphs("RNN")

        # self.model_compile(model_name="LSTM", value_to_monitor="val_loss", min_delta_value=0.1, mode_option="min", starting_epoch=60, optimizer_name="adam", num_of_epochs=200)
        # self.save_model_and_graphs("LSTM")

        # self.model_compile(model_name="GRU", value_to_monitor="val_loss", min_delta_value=0.1, mode_option="min", starting_epoch=50, optimizer_name="adam", num_of_epochs=175)
        # self.save_model_and_graphs("GRU")

        # self.k_fold_cross_validation(model_name="RNN", value_to_monitor="loss", min_delta_value=0.3, mode_option="min", starting_epoch=110, optimizer_name="adam", num_of_splits=5, num_of_epochs=150)
        # self.k_fold_cross_validation(model_name="LSTM", value_to_monitor="loss", min_delta_value=0.3, mode_option="min", starting_epoch=60, optimizer_name="adam", num_of_splits=5, num_of_epochs=200)
        # self.k_fold_cross_validation(model_name="GRU", value_to_monitor="loss", min_delta_value=0.3, mode_option="min", starting_epoch=50, optimizer_name="adam", num_of_splits=5, num_of_epochs=175)

    def get_data_from_directory(self):
        signs_labels_enum = {label:num for num, label in enumerate(self.sign_folders)}

        print("Signs and their indexes: ", signs_labels_enum)

        sign_data_collection = []
        signs_labels = []

        last_time = datetime.now()

        for data_folder in self.sign_folders: # For each data folder in array of folder names
            print(datetime.now(), ": Processing ", data_folder, " (previous folder processed in {})".format(datetime.now() - last_time)) # For debugging
            last_time = datetime.now()
            for sub_folder in os.listdir(os.path.join(self.directory, data_folder)): # For each subfolder in the folder of the sign_folders
                sign_data = [] # Array to store the values
                invalid_data = False

                for frame in range(0, self.max_frame_amount): # From 1st until maximum allowed frame
                    current_file = os.path.join(self.directory, data_folder, sub_folder, "{}.npy".format(frame)) # From each .npy file

                    if (os.path.isfile(current_file) and os.path.getsize(current_file) > 0):
                        pass 
                        # !! Comment once finished gathering the data
                        # file_data = np.load(current_file) # Get info from .npy file
                        # sign_data.append(file_data) # And append it to the window
                    else: 
                        invalid_data = True

                if invalid_data == False: # If there are data in the array
                    sign_data_collection.append(sign_data) # Append data to sequence of defined letter or word
                    signs_labels.append(signs_labels_enum[data_folder]) # Append name of the folder along with its index

        # !! Comment once finished gathering the data
        # joblib.dump(sign_data_collection, 'sign_data_full.joblib') # Save data to stop endless pointless extraction upon each start 

        print(datetime.now(), ": Processing finished", "(last folder processed in {})".format(datetime.now() - last_time)) # For debugging

        sign_data = joblib.load('sign_data_full.joblib')

        self.X_data = np.array(sign_data) # 3D array of all data from frames. [file[frames[data]]]
        self.Y_data = to_categorical(signs_labels) # 2D array for categories in both 

    def create_model(self, model_name: str) -> Sequential:
        """Method to create a Keras Machine Learning model.\n
            Returns Keras model of specified Machine Learning algorithm.

            Keyword arguments:\n
            model_name -- name of supervised machine learning algorithm fitted for sequential data classification.\n
        """
        model = None
        
        if(model_name == "RNN"):
            model = Sequential([
                Input(shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))),
                SimpleRNN(48, return_sequences=True, activation='relu'),
                GaussianNoise(0.15),
                SimpleRNN(32, return_sequences=False, activation='relu'),
                Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
            ])
        elif (model_name == "LSTM"):
            model = Sequential([
                Input(shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))), # 1530 - non-traceable, 1710 - traceable
                LSTM(128, return_sequences=True, activation='sigmoid'),
                GaussianNoise(0.2),
                LSTM(128, return_sequences=True, activation='sigmoid'),
                Dropout(0.2),
                LSTM(128, return_sequences=False, activation='sigmoid'),
                Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
            ])
        elif (model_name == "GRU"):
            model = Sequential([
                Input(shape=(self.max_frame_amount, (self.face_points_amount + 2 * self.hand_points_amount))), # 1530 - non-traceable, 1710 - traceable
                GRU(128, return_sequences=True, activation='sigmoid'),
                GaussianNoise(0.25),
                GRU(128, return_sequences=False, activation='sigmoid'),
                Dense(np.array(self.sign_folders).shape[0], activation='softmax') # Layer that contains all possible outputs
            ])
        else:
            exception_msg = "There is no types of ML models for '{}'. Check definition and try again.".format(model_name)
            raise ValueError(exception_msg)

        return model

    def model_compile(self, model_name: str, value_to_monitor: str, min_delta_value: float, mode_option: str, starting_epoch: int, optimizer_name: str, num_of_epochs: int):
        """Method to compile a Keras Machine Learning model.\n

            Keyword arguments:\n
            model_name -- name of supervised machine learning algorithm fitted for sequential data classifcation.\n
            value_to_monitor -- name of the value that needs to be tracked during Machine Learning process.\n
            min_delta_value -- amount of delta needed to account for valid progress.\n
            mode_option -- option of the mode (what to track, to min or to max value progress).\n
            starting_epoch -- from which epoch start EarlyStopping.\n
            optimizer_name -- name of the optimizer used in Machine Learning.\n
            num_of_epochs -- number of epochs for Machine Learning process.\n
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=self.test_size)
        
        self.model = self.create_model(model_name)

        early_stop = EarlyStopping(
            monitor=value_to_monitor,
            min_delta=min_delta_value,
            patience=15,
            verbose=0,
            mode=mode_option,
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=starting_epoch
        )

        self.model.compile(optimizer=optimizer_name, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train, epochs=num_of_epochs, validation_data=(self.x_test, self.y_test), callbacks=[early_stop])

        self.model.summary()

    def save_model_and_graphs(self, model_name: str):
        """Method to save a Keras Machine Learning model and its graphs (learning curves, confusion matrix and model architecture).\n

            Keyword arguments:\n
            model_name -- name of supervised machine learning algorithm fitted for sequential data classification.\n
        """
        # Accuracy graph
        plt.plot(self.history.history['categorical_accuracy'], color='#0277bd')
        plt.plot(self.history.history['val_categorical_accuracy'], linestyle='dashed', color='#f77500')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.title('{} model categorical accuracy overtime (test_size - {})'.format(model_name, self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('categorical_accuracy')
        plt.grid()
        plt.savefig("{} - Accuracy (test_size - {}).png".format(model_name, self.test_size))
        plt.clf()

        # Loss graph
        plt.plot(self.history.history['loss'], color='#0277bd')
        plt.plot(self.history.history['val_loss'], linestyle='dashed', color='#f77500')
        plt.legend(['loss', 'val_loss'])
        plt.title('{} model loss overtime (test_size:{})'.format(model_name, self.test_size))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig("{} - Loss (test_size - {}).png".format(model_name, self.test_size))
        plt.clf()

        ypred = self.model.predict(self.x_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        ypred = np.argmax(ypred, axis=1).tolist()
        conf_matrix = confusion_matrix(ypred, ytrue)

        # Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(15, 15))
        conf_matrix_figure = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.signs_to_learn)
        conf_matrix_figure.plot(cmap=plt.cm.YlOrRd, ax=ax)
        plt.title('{} model confusion matrix (test_size - {})'.format(model_name, self.test_size))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="default")
        plt.savefig("{} - Confusion Matrix (test_size - {}).png".format(model_name, self.test_size))
        plt.clf()

        plot_model(model=self.model, to_file='{}_model_structure.png'.format(model_name), show_shapes=True)
        save_model(self.model, '{}_model_({}).keras'.format(model_name, self.test_size))

        fig, ax = plt.subplots(figsize=(8, 6))
 
    def k_fold_cross_validation(self, model_name: str, value_to_monitor: str, min_delta_value: float, mode_option: str, starting_epoch: int, optimizer_name: str, num_of_splits: int, num_of_epochs: int):
        """Method to perform k-fold cross validation on Keras Machine Learning model.\n

            Keyword arguments:\n
            model_name -- name of supervised machine learning algorithm fitted for sequential data classifcation.\n
            value_to_monitor -- name of the value that needs to be tracked during Machine Learning process.\n
            min_delta_value -- amount of delta needed to account for valid progress.\n
            mode_option -- option of the mode (what to track, to min or to max value progress).\n
            starting_epoch -- from which epoch start EarlyStopping.\n
            optimizer_name -- name of the optimizer used in Machine Learning.\n
            num_of_splits -- number of splits for data during k-fold cross validation. Also specifies a number how many times cross validation will be performed with different set of validation data sets.\n
            num_of_epochs -- number of epochs for Machine Learning process.\n
        """

        early_stop = EarlyStopping(
            monitor=value_to_monitor,
            min_delta=min_delta_value,
            patience=10,
            verbose=0,
            mode=mode_option,
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=starting_epoch
        )


        kfold = KFold(n_splits=num_of_splits, shuffle=True)
        k_fold_num = 0

        for train, test in kfold.split(self.X_data, self.Y_data):
            self.model = self.create_model(model_name)
            self.model.compile(optimizer=optimizer_name, loss='categorical_crossentropy', metrics=['categorical_accuracy', Precision(), Recall(), F1Score(average="weighted"), AUC()])

            self.model.fit(self.X_data[train], self.Y_data[train], epochs=num_of_epochs, callbacks=[early_stop])

            self.history = self.model.evaluate(self.X_data[test], self.Y_data[test], return_dict=True)

            if k_fold_num == 0:
                accuracy_score = self.history['categorical_accuracy']
                loss_score = self.history['loss']
                precision_score = self.history['precision']
                recall_score = self.history['recall']
                f1score_score = self.history['f1_score']
                auc_score = self.history['auc']
            else:
                accuracy_score = self.history['categorical_accuracy']
                loss_score = self.history['loss']
                precision_score = self.history['precision_{}'.format(k_fold_num)]
                recall_score = self.history['recall_{}'.format(k_fold_num)]
                f1score_score = self.history['f1_score']
                auc_score = self.history['auc_{}'.format(k_fold_num)]

            headers = ["Accuracy", "Loss", "Precision", "Recall", "F1", "AUC ROC", "Date"]

            with open('ML_Results_{}.csv'.format(model_name), 'a') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                if os.stat('ML_Results_{}.csv'.format(model_name)).st_size == 0:
                    writer.writeheader()
                writer.writerows([{'Accuracy': accuracy_score, 'Loss': loss_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1score_score, 'AUC ROC': auc_score, 'Date': datetime.now()}])

            k_fold_num += 1

    def count_files(self, directory) -> int:
        total_file_amount = 0

        for file in os.scandir(directory):
            if file.is_file():
                total_file_amount += 1

        return total_file_amount

if __name__ == "__main__":
    model_creator = ModelCreator()