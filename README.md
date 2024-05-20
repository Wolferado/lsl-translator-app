# Latvian Sign Language Translator App

> [!IMPORTANT]
> This app is developed for this author bachelor's thesis on “Real-time computer vision solution for Latvian sign language recognition”.
>
> Link to the bachelor's thesis: *add once uploaded*

> [!NOTE]
> This repository contains code for the Python app that manages to recognize 44 signs (33 letters, 10 words, 1 empty sign) from Latvian Sign Language by using Machine Learning.

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

## Machine Learning models' statistics

TODO: Add information about stats

## Used technologies and documentation:

- MediaPipe Hands - for hands recognition and features extraction 
- MediaPipe Face Mesh - for face recognition and features extraction
- Keras - for creating Machine Learning models
- Flet - for User Interface
- NumPy - for mathematical operations with data
