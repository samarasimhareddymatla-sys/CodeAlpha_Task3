This project involves building a machine learning model that can classify handwritten characters (either digits or alphabets). The MNIST dataset (for digits 0-9) is the most widely used dataset for such tasks, but the EMNIST dataset can be used for alphabets. The model built for this task is typically a Convolutional Neural Network (CNN) due to its efficiency in handling image data. The model is trained to classify images of handwritten characters into different categories (e.g., digits or letters).
Tech Stack:
TensorFlow and Keras for building deep learning models, especially CNNs.
OpenCV or Pillow for image preprocessing (resizing, grayscale conversion).
Streamlit for creating a user interface where users can upload or draw handwritten digits, and the model will predict them.
Key Steps:
Preprocess the dataset (resizing images to 28x28, normalizing pixel values, etc.).
Design and train a CNN to classify the images.
Evaluate the model on the test set (MNIST or EMNIST) for accuracy.
Save the trained model and load it in a Streamlit app to allow users to either upload a digit or draw on a canvas and get the prediction.
