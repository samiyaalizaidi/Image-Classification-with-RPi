# CPE 4903: Neural Networks and Machine Learning 
This project aims to train two Convolutional Neural Networks (CNNs): 'Hand Digit Recognition' and 'Cats and Dogs Classification.' The models will then be deployed on a Raspberry Pi to perform inference through PiCamera. 

The Hand Digit Classifier was required to have an accuracy of $`\geq 99\%`$ whereas the Cats and Dogs Classification model must have an accuracy of $`\geq 85\%`$.


## Design and Implementation

### Hardware Specifications
Both models were deployed on the same device. The products and their specifications are:
<ul>
    <li>Raspberry Pi 4 - Model B - 2GB RAM - 64-bit quad-core Cortex-A72 Processor</li>
    <li>Raspberry Pi Camera Module - imx219</li>
</ul>

### Software Specifications
The software specifications along with the Python libraries are:
<ul>
    <li>Raspbian 32-bit OS, Bookworm</li>
    <li>Python 3.7.12</li>
    <li>TensorFlow 2.4.0</li>
    <li>Keras Preprocessing</li>
    <li>OpenCV</li>
    <li>Pyenv for creating a local environment for Python</li>
</ul>

### Training the Models

#### Hand-Digit Recognition

The dataset used for training the model MNIST. The dataset was split into training and testing, with 60000 images for training and 10000 images for testing images. The size of the input images was $`(28, 28, 1)`$.

The model was trained using TensorFlow and Keras on Google Colab. It's a custom model consisting of 4 convolutional layers and 3 dense layers. Max pooling was also used after every convolutional layer. 
