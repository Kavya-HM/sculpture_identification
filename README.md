# Historical Sculpture detection on Raspberry Pi
This project uses Convolutional Neural Networks (CNN) to recognize different types of Historical sculptures.
The CNN model is trained on a dataset of images of sculptures from various temples in Karnataka.

# Dataset
The dataset consists of images of sculptures from five different categories:

Garuda
Madanikas
Nandi
Narashimha
Vishnu

# Model
The model used in this project is a custom convolutional neural network (CNN) built using the TensorFlow and Keras libraries.
The architecture of the model consists of three convolutional layers, each followed by a max pooling layer, and two fully connected (dense) layers. The activation function used in all layers except the output layer is ReLU, and the output layer uses the softmax activation function to produce the final predictions. The model was trained on a dataset of temple sculptures using the Adam optimizer and categorical cross-entropy loss.
The final trained model achieved an accuracy of around 95% on the test set. The model was saved in HDF5 format and can be loaded using the tf.keras.models.load_model() function in TensorFlow.

# Usage
To use the trained model for recognizing sculptures in new images, follow these steps:

1. Install the required libraries (tensorflow, numpy, opencv, and matplotlib)
2. Load the trained model using tf.keras.models.load_model()
3. Load an image using cv2.imread()
4. Preprocess the image (resize it to match the size of the images used for training and normalize the pixel values)
5. Add a batch dimension to the image (since the model expects a batch of images)
6. Make a prediction using the trained model (model.predict())
7. Display the predicted class label along with the corresponding temple name and history (from a CSV file)

# Results
The trained model achieves an accuracy of approximately 96.39% on the validation set.
We also tested the model on several new images of sculptures and found that it performs well in recognizing different types of sculptures.

# Future Work
Possible future improvements to this project include:

* Collecting a larger and more diverse dataset of Indian sculptures
* Trying out different pre-trained models for transfer learning
* Fine-tuning the hyperparameters of the model to improve its accuracy
