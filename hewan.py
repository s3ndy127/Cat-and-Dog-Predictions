import tensorflow as tf
import tensorflow_datasets as tfds
import streamlit as st
from PIL import Image
import math

st.set_page_config(
    page_title="Cat & Dog Prediction",
    layout="wide"
)

# Load the dataset
dataset, info = tfds.load('oxford_iiit_pet', split='train+test', with_info=True, as_supervised=True)

train_ratio = 0.8
test_ratio = 0.2
train_size = int(len(dataset) * train_ratio)
test_size = int(len(dataset) * test_ratio)

train_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
test_dataset = remaining_dataset.take(test_size)

# Preprocess data
IMG_SIZE = 224
batch_size = 32

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess_image).batch(batch_size)
test_dataset = test_dataset.map(preprocess_image).batch(batch_size)

model = tf.keras.models.load_model('model_pet_mobilenetv2.h5')

def preprocess_single_image(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def predict_image(image):
    preprocessed_image = preprocess_single_image(image)
    expanded_image = tf.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(expanded_image)
    predicted_class_index = tf.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    return predicted_class_index, confidence

# Streamlit app
st.title("Cat & Dog Prediction")

# Image upload and prediction 
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    predicted_class_index, confidence = predict_image(image)
    class_names = info.features['label'].names
    predicted_class = class_names[predicted_class_index]

    # Create two columns to display the image and prediction
    col1, col2 = st.columns(2)
    
    # Display the image in the first column
    col1.image(image, caption="Uploaded Image", use_column_width=True)

    # Display the prediction and confidence in the second column with style
    if confidence >= 0.75:
        col2.write(f"<p style='text-align:center;font-size:26px;font-weight:bold;'>Predicted class: {predicted_class}</p>", unsafe_allow_html=True)
        col2.write(f"<p style='text-align:center;font-size:26px;font-weight:bold;'>Confidence: {confidence:.4f}</p>", unsafe_allow_html=True)

    # Display similar images
    
        class_dataset = remaining_dataset.filter(lambda _, label: tf.equal(label, predicted_class_index))
        sample_images = list(class_dataset.as_numpy_iterator())

        # Calculate the number of rows and columns based on the number of images
        num_images = len(sample_images)
        num_columns = 3
        num_rows = math.ceil(num_images / num_columns)

        # Iterate over the images and display them in Streamlit
        col3, col4, col5 = st.columns(3)
        for i, (image, label) in enumerate(sample_images):
            image_np = image

            # Determine the row and column index for the current image
            row = i // num_columns
            col = i % num_columns

            # Display the image
            if col == 0:
                col3.image(image_np, use_column_width=True)
            elif col == 1:
                col4.image(image_np, use_column_width=True)
            else:
                col5.image(image_np, use_column_width=True)
    else:
        col2.write("<p style='text-align:center;font-size:20px;font-weight:bold;'>Nilai confidence kurang dari 0.75, jadi tidak dapat menampilkan gambar yang serupa.</p>", unsafe_allow_html=True)
