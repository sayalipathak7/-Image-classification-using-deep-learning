The aim of this assignment is to classify input images into one of the following categories: buildings, forest, glacier, mountain, sea, and street.

I am using PyTorch framework to solve this problem of image classification using deep learning.

I am using dataset from Kaggle - Image Classification Dataset. It includes total of 17034 (training+testing) images, categorized into 6 classes: buildings, forest, glacier, mountain, sea, and street.

Code explanation -

Installing necessary libraries like torch, torchvision, matplotlib, and Pillow.

Downloading dataset which includes images from Kaggle website.

Setting paths to training and testing folders.

Transforming all the input images to a consistent size of 128x128 pixels.

Visualizing the dataset -> Showing orignal and transformed images with the class.

Analyzing test data and training data by plotting graph.

Creating CNN (Convolution Neural Network) model - The model consists of two convolutional blocks, each followed by a ReLU activation function and max-pooling layers. The final classification is performed through a fully connected layer.

Model description

Convolution blocks - The cnn model constatis of 2 convolution blocks. Each contains multiple convolution layers with kernel size 3*3 and ReLU activation function.
These blocks are used to extract important features from input images.

Max pooling layers - Max pooling layer with a kernel size of 2*2 is applied after each convolution layer. It is used to reduce spatial dimensions and retain important information.

Fully connected layer - This layer takes the flattened output from the convolutional blocks and maps it to the number of classes in the dataset. The classification of the images is done by this layer.

Training, testing, and visualization - The model is trained over 5 epochs using the Adam optimizer and the cross-entropy loss function to monitor and optimize model's performance. I am monitoring training progress through loss and accuracy metrics for training and test dataset and visualizaing the result using Tensor Board. From loss and accuracy metrics, I got to know how the model is learning over epochs. I could see decreasing trend in loss and an increasing trend in accuracy for both training and testing datasets

Inference - This model is used to classify verious images into given classes. I am providing visual assessment of model's performance by showing predicted class vs actual class.
