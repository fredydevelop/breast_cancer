import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
import random
import os
import imghdr
import streamlit as st
import pickle as pk
import cv2
import requests
from PIL import Image
from io import BytesIO
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model, save_model
from PIL import Image
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

col1, col2 = st.columns(2)

with col1:
    st.image("clarascan_logo.jpg")
    st.header("Breast Guard AI")






def insert():
    st.header("Upload Image to detect ")
    

    # File uploader widget
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png","bmp"], key="upl")

    if uploaded_file is not None:
        # Convert the uploaded image to RGB
        img = Image.open(uploaded_file).convert("L")   # grayscale
        img = np.array(img)                            # shape: (H, W)

        img = np.expand_dims(img, axis=-1)             # shape: (H, W, 1)
    
        resize_img = tf.image.resize(img, (256, 256))  # now valid
    
        img_array = np.expand_dims(resize_img, axis=0) # shape: (1, 256, 256, 1)
        img_array = img_array / 255.0
        
        # To load the model
        loaded_model = tf.keras.models.load_model("new_breastcancer_model.keras")        # Make the prediction
        
        if st.button("Predict"):
            prediction = loaded_model.predict(img_array, verbose=0)
        
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_labels = ['benign', 'malignant']
            predicted_category = class_labels[predicted_class]
            confidence = float(prediction[0][predicted_class]) * 100
        
            st.image(img, caption="Uploaded image")
            st.write("Raw prediction:", prediction)
        
            if predicted_category == "malignant":
                st.error(
                    f"The model predicts {predicted_category} with {confidence:.2f}% confidence. "
                    f"This may indicate a cancerous condition. Please consult a medical professional."
                )
            else:
                st.success(
                    f"The model predicts {predicted_category} with {confidence:.2f}% confidence. "
                    f"This may indicate a non-cancerous condition."
                )


# def all_predict():
#     st.title("Upload item to predict")
#     # File uploader widget
#     uploaded_file = st.file_uploader("select the folder containing your image", key="jacking")

#     if uploaded_file is not None:
#         print("")



# if selection == "Predict a Single Item":
#     main()

# if selection == "Predict for Multi-Patient":
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     #---------------------------------#
#     # Prediction
#     #--------------------------------
#     #---------------------------------#
#     # Sidebar - Collects user input features into dataframe
#     st.header('Upload your image file here')
#     uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"],key="jjj")
#     #--------------Visualization-------------------#
#     # Main panel
    
#     # Displays the dataset
#     # if uploaded_file is not None:
#     #     #load_data = pd.read_table(uploaded_file).
#     #     multi(uploaded_file)
#     # else:
#     #     st.info('Upload your dataset !!')



#insert()


if __name__ == "__main__":
    insert()

