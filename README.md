# Latvian Sign Language Translator App

> [!WARNING]
> This project has been archived due to it being presented and evaluated.

> [!IMPORTANT]
> This app is developed for this author bachelor's thesis on “Real-time computer vision solution for Latvian sign language recognition”.

> [!NOTE]
> This repository contains code for the Python app that manages to recognize 44 signs (33 letters, 10 words and 1 empty sign) from Latvian Sign Language by using Machine Learning.

## Overview
App contains 4 views with their own purpose:

- Home View (greet user and show useful tips);
- Recognition View (recognize showed signs and give their meaning);
- Data Extractor View (extract data about the sign from the video);
- Data Creator View (create a video needed for data extraction process).

## Recognition logic and working principle

Recognition View contains so called Recognition module that needs to be initiated. Once loaded, UI feed will show current videocamera video stream along with textfield and some UI options.

Upon starting, a 40 frame-length video will be recorded. Once 40th frame is reached, last 30 frames will be used for sign recognition process using Machine Learning models.

Machine learning models are created with Keras library and use following algorithms in their structure (one per each model):

- RNN (Recurrent Neural Network);
- LSTM (Long-Short Term Memory);
- GRU (Gated Recurrent Unit).

## Showcase

![GIF of showcasing the functionality of the app](/materials/LSL%20-%2010%20sec.gif)

## Machine Learning models statistics

### RNN 5-fold Cross-Validation

| Fold Nr | Accuracy | Loss | Precision | Recall | F1 |
| ------- | -------- | ---- | --------- | ------ | -- |
|#1       | 0,494    | 1,676| 0,601     | 0,401  | 0,487|
|#2       | 0,673    | 0,983| 0,779     | 0,582  | 0,664|
|#3       | 0,874    | 0,464| 0,925     | 0,838  | 0,874|
|#4       | 0,563    | 1,336| 0,861     | 0,281  | 0,538|
|#5       | 0,774    | 0,716| 0,844     | 0,701  | 0,770|
| **Average** | 0,676    | 1,035| 0,802     | 0,561  | 0,66|

### LSTM 5-fold Cross-Validation

| Fold Nr | Accuracy | Loss | Precision | Recall | F1 |
| ------- | -------- | ---- | --------- | ------ | -- |
|#1       | 0,894    | 0,310| 0,920     | 0,863  | 0,893|
|#2       | 0,871    | 0,419| 0,893     | 0,849  | 0,868|
|#3       | 0,902    | 0,287| 0,931     | 0,874  | 0,901|
|#4       | 0,900    | 0,305| 0,927     | 0,882  | 0,896|
|#5       | 0,991    | 0,288| 0,952     | 0,864  | 0,911|
| **Average** | 0,896    | 0,322| 0,925     | 0,866  | 0,894|

### GRU 5-fold Cross-Validation

| Fold Nr | Accuracy | Loss | Precision | Recall | F1 |
| ------- | -------- | ---- | --------- | ------ | -- |
|#1       | 0,933    | 0,219| 0,944     | 0,926  | 0,933|
|#2       | 0,945    | 0,170| 0,958     | 0,933  | 0,945|
|#3       | 0,952    | 0,159| 0,970     | 0,940  | 0,951|
|#4       | 0,949    | 0,166| 0,965     | 0,941  | 0,949|
|#5       | 0,936    | 0,206| 0,947     | 0,930  | 0,937|
| **Average** | 0,943    | 0,184| 0,957     | 0,934  | 0,943|

## Used technologies and documentation:

- MediaPipe Hands - for hands recognition and features extraction [(hands.md)](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)
- MediaPipe Face Mesh - for face recognition and features extraction [(face_mesh.md)](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#python-solution-api)
- Keras - for creating Machine Learning models [(Documentation)](https://keras.io/)
- Flet - for creating User Interface [(Documentation)](https://flet.dev/)
- NumPy - for mathematical operations with data [(Documentation)](https://numpy.org/)
