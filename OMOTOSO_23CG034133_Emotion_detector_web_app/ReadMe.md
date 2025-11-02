## Overview

This web application detects a person's **emotion** either from an uploaded image or a live webcam capture.

---

The project was built using **Python** for the web app and model training, and **Streamlit** for hosting. It logs each user's activity (name, image, and prediction result) into a local **SQLite** database

## Features

Detects emotions from:

- Uploaded image files
- Real time webcam input

Displays:

- Detected dominant emotion
- Confidence scores for all emotions.
- Saves user information and predictions to a local database.

Includes both:

- A custom CNN model. (You can find it here: )
- A deepFake fallback model (Just in case the CNN model has issues.)

## How to run it

To test it out lon your own machine:

- Clone the repo:
  git clone https://github.com/oluwatomi-omotoso/emotion-detector-app.git

- So you don't break anything, create a virtual environment
- And then install the dependencies:

  pip install -r requirements.txt

**OR**

You could just try out the live web app I made, you can find it here:

## Optional

You could train your own model for improved resuls.

Here's the dataset I used it for mine: https://www.kaggle.com/datasets/msambare/fer2013
