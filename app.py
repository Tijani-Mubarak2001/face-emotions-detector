import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os

# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# face Expression Detection function
def face_expression():

    cover_img = Image.open('face recog.jpg')
    st.image(cover_img, caption='Face Expression Recog', use_column_width=True)
    st.title("Face Emotion Detector")
    st.write("""
            Face detection is a fundamental and extensively studied problem in computer vision. It is the first and primary step for many applications which include face recognition/verification, face tracking, facial behavior analysis, and many other applications. A major challenge for face detector is to detect faces in unconstrained conditions (also called in-the-wild) such as variations in pose, illumination, scale, expressions, makeup, and occlusion. Recently, accuracy and performance of face detectors have improved tremendously because of the use of Convolutional Neural Network (CNN).
            ----
            """)
    st.write(
        '''
        **Haar Cascade** is an object detection algorithm.
        It can be used to detect objects in images or videos. 

        The algorithm has four stages:

            1. Haar Feature Selection 
            2. Creating  Integral Images
            3. Adaboost Training
            4. Cascading Classifiers 
            ''')
    st.write("""
            This CNN model deployed classifies the images based on Emotions detected on faces in the picture, by learning from 28000 different images from Kaggle
            ----
            ### Briefly, this is how the CNN model learnt to classify the images
            - Tensorflow, Keras ImageDataGenerator, os, cv2 Libraries were imported
            - Input images with Human faces on it are read.
            - If the Input Image is color (RGB), then it is converted to Gray scale Image and the pixel values is saved to a 2D array. Else save the pixel values of the input image to a 2D array.
            - Training images was processed, rescaled and normalized with ImageDataGenerator into dimensions suited for training with the CNN model and fature maps generated
            - The CNN model was initialized
            - A Rectifier Linear Unit function is applied to increase non-linearity
            - pooling is then applied to get the details in each feature map in other to concentrate on the important features
            - A Second Convolutional layer is added using the Rectifier Linear Unit function
            - Then a final output Layer is created with the Softmax activation  function
            - The result is flattened i.e compressed into a single Vector
            - The vector is then fed to a fully connected artificial Neural Network.
            - Features are picked up by the layers of the Neural Network.
            - weights are adjusted through the method of backward propagation so as to minimize the loss function
            - After series of back propagation and forward propagation specified by the numbers of epochs, the layers learnt enough to classify images based on the Emotions detected on each faces.
            ----

            #### Test the model :sunglasses:
            """)
    # Loading images
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp', 'jfif'])
    # Initializing face_detect and face_roi array
    face_detect = 0
    face_roi = np.array([])
    pred_cat = ""

    if image_file is not None:
        image = Image.open(image_file)
        # Converting Image to Gray Scale image
        image = np.array(image.convert('RGB'))
        shape = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Detecting faces on image

        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + w), (255, 0, 0), 2)
            facess = face_cascade.detectMultiScale(roi_gray)
            eyes = eye_cascade.detectMultiScale(roi_color)
            # Drawing a rectangle over detected faces
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Checking the number of faces detected
            if len(faces) == 0:
                face_detect = 0
            elif len(faces) >= 1:
                face_detect = len(faces)
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Shape:" + str(shape))

        if face_detect == 0:
            st.warning("No face detected")
        elif face_detect > 0:
            st.success("Found {} face/faces\n".format(len(faces)))
            if st.button("Predict"):
                if face_roi.size == 0:
                    st.warning("Unable to predict on picture :worried: ")
                else:
                    final_image = cv2.resize(face_roi, (224, 224))  # Resizing image into suitable prediction form
                    final_image = np.expand_dims(final_image, axis=0)
                    final_image = final_image/255.0  # Normalizing image
                    loaded_model = tf.keras.models.load_model('correct_model')  # Loading saved model
                    predictions = loaded_model.predict(final_image)
                    if np.argmax(predictions) == 0:
                        pred_cat = "Angry"
                    elif np.argmax(predictions) == 1:
                        pred_cat = "Disgust"
                    elif np.argmax(predictions) == 2:
                        pred_cat = "Fear"
                    elif np.argmax(predictions) == 3:
                        pred_cat = "Happy"
                    elif np.argmax(predictions) == 4:
                        pred_cat = "Neutral"
                    elif np.argmax(predictions) == 5:
                        pred_cat = "Sad"
                    elif np.argmax(predictions) == 6:
                        pred_cat = "Surprise"
                    st.write(f'''## Prediction: {pred_cat}''')
                st.write('''Check Out My Pet Classifier and About Page On the Navigation Bar''')

def about():
    cover_img = Image.open('profile picture.jpg')
    st.image(cover_img, caption='Profile Picture', use_column_width=True)
    st.title("I am Tijani Mubarak Adewale")
    st.write("""
    ----
    I am glad you took your time to go through my App.
    My AI Models are not the most accurate, So, if your images are classified wrongly and you want to make known some
    observations, please contact me by sending a mail to Tijanimubarak2001@gmail.com.
    You can also contact me through my website [![Tijani Mubarak](https://img.shields.io/badge/Author-@TijaniMubarak-gray.svg?colorA=gray&colorB=dodgergreen&logo=react)](https://tijaniportfolio.netlify.app/) Linkedin
    [![Tijani Mubarak](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white)](https://www.linkedin.com/in/mubaraktijani/) and Twitter  [![Tijani Mubarak](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=gray)](https://twitter.com/TijaniMubarakA1)  
    
    """)


def dog_cat_classifier():
    cover_img = Image.open('Cat_Dog_Cover.jpg')
    st.image(cover_img, caption='Cat VS Dog', use_column_width=True)
    st.title("Pets Classifier")

    st.write("""
    This CNN model deployed basically classifies the images Of Dogs and Cats, by learning from 8000 different images of cats and dogs
    ----
    ### How this CNN model learnt to classify the images
    - Tensorflow, Keras and ImageDataGenerator Libraries were imported
    - Training images was processed and rescaled with ImageDataGenerator into dimensions suited for training with the CNN model and fature maps generated
    - The CNN model was initialized
    - A Rectifier Linear Unit function is applied to increase non-linearity
    - pooling is then applied to get the details in each feature map in other to concentrate on the important features
    - A Second Convolutional layer is added using the Rectifier Linear Unit function
    - Then a final output Layer is created with the sigmoid function
    - The result is flattened i.e compressed into a single Vector
    - The vector is then fed to a fully connected artificial Neural Network.
    - Features are picked up by the layers of the Neural Network.
    - weights are adjusted through the method of backward propagation so as to minimize the loss function
    - After series of back propagation and forward propagation specified by the numbers of epochs, the layers would have learnt enough to classify images such as that of Cats and Dogs.
    ----

    #### Test the model :sunglasses:
    """)
    image_file = st.file_uploader("Upload image of a Cat or Dog", type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
        image_ = Image.open(image_file)

        st.image(image_, caption='Uploaded Image.', use_column_width=True)
        test_image = image_.resize((64, 64))

        # Converting image to acceptable form (numpy array)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        loaded_model = tf.keras.models.load_model('the_cnn_model')

        if st.button("Predict"):
            dogQuotes = ['A dog is the only thing on earth that loves you more than you love yourself.',
                         "No matter how you're feeling, a little dog gonna love you.",
                         "Dog is God spelled backward.",
                         "If you want someone to love you forever, buy a dog, feed it and keep it around.",
                         "Money can buy you a fine dog, but only love can make him wag his tail.",
                         "The love of a dog is a pure thing. He gives you a trust which is total. You must not betray it.",
                         "You know, a dog can snap you out of any kind of bad mood that you’re in faster than you can think of.",
                         "Dogs do speak, but only to those who know how to listen."]
            catQuotes = ["Cats are connoisseurs of comfort.",
                         "You can not look at a sleeping cat and feel tense.",
                         "Kittens are angels with whiskers.",
                         "Cats leave paw prints in your heart, forever and always.",
                         "Like all pure creatures, cats are practical.",
                         "Heaven will never be Paradise unless my cats are there waiting for me.",
                         "Cats choose us; we don’t own them.",
                         "There are few things in life more heartwarming than to be welcomed by a cat."]
            import random
            result = loaded_model.predict(test_image)
            n = random.randint(0, 7)
            if result[0][0] == 1:
                prediction = 'Dog'
                Quotes = dogQuotes[n]
            else:
                prediction = 'Cat'
                Quotes = catQuotes[n]
            st.write(f''' ## Prediction: {prediction}
                    {Quotes} ''')
            st.subheader("Note: Model Accuracy is 79%")





def main():

    activities = ["Face Emotion Recognition", "Pets classifier", "About"]
    choice = st.sidebar.selectbox("Navigate to pages of interest", activities)

    if choice == "Face Emotion Recognition":
        face_expression()

    elif choice == "Pets classifier":
        dog_cat_classifier()
    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
