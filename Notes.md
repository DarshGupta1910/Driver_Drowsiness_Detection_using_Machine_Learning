[driver drowsiness detection blog](https://towardsdatascience.com/drowsiness-detection-with-machine-learning-765a16ca208a)
# Data preparation

###  os module
The OS module in Python provides functions for interacting with the operating system. OS comes under Python’s standard utility modules. This module provides a portable way of using operating system-dependent functionality. The *os* and *os.path* modules include many functions to interact with the file system.

### shutil 
The shutil in Python is a module that offers several functions to deal with operations on files and their collections. It provides the ability to copy and removal of files.

    shutil.copy("file_path", "path_to_directory_where_you_want_to_copy_to")

### glob
is used to **find the files and folders whose names follow a specific pattern**. The searching rules are similar to the Unix Shell path expansion rules.

### tqdm 
is used to **find the files and folders whose names follow a specific pattern**. The searching rules are similar to the Unix Shell path expansion rules.

# Model Training
 ### Tensorflow 
 TensorFlow is a software library or framework, designed by the Google team to implement machine learning and deep learning concepts in the easiest manner. It combines the computational algebra of optimization techniques for easy calculation of many mathematical expressions.
-   It includes a feature of that defines, optimizes and calculates mathematical expressions easily with the help of multi-dimensional arrays called tensors.
    
-   It includes a programming support of deep neural networks and machine learning techniques.
    
-   It includes a high scalable feature of computation with various data sets.
    
-   TensorFlow uses GPU computing, automating management. It also includes a unique feature of optimization of same memory and the data used.

### Keras 
Keras is a high level deep learning API written in python for neural networks. It supports multiple backend neural networks computations and makes implementing neural networks easy.

### Inception V3
CNN Architecture 
Trained on ImageNet Data Set. 
Inception-v3 is a convolutional neural network that is 48 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 299-by-299.

The architecture of an Inception v3 network is progressively built, step-by-step, as explained below:
**1. Factorized Convolutions:**  this helps to reduce the computational efficiency as it reduces the number of parameters involved in a network. It also keeps a check on the network efficiency.
**2. Smaller convolutions:**  replacing bigger convolutions with smaller convolutions definitely leads to faster training. Say a 5 × 5 filter has 25 parameters; two 3 × 3 filters replacing a 5 × 5 convolution has only 18 (3*3 + 3*3) parameters instead.
**3. Asymmetric convolutions:** A 3 × 3 convolution could be replaced by a 1 × 3 convolution followed by a 3 × 1 convolution. If a 3 × 3 convolution is replaced by a 2 × 2 convolution, the number of parameters would be slightly higher than the asymmetric convolution proposed.
**4. Auxiliary classifier:** an auxiliary classifier is a small CNN inserted between layers during training, and the loss incurred is added to the main network loss. In GoogLeNet auxiliary classifiers were used for a deeper network, whereas in Inception v3 an auxiliary classifier acts as a regularizer.
**5. Grid size reduction:** Grid size reduction is usually done by pooling operations. However, to combat the bottlenecks of computational cost, a more efficient technique is proposed:

![See the source image](https://www.researchgate.net/publication/341563435/figure/download/fig3/AS:941464410402818@1601474015137/Block-diagram-of-Inception-v3-improved-deep-architecture.png)
 
 ### Transfer Learning
 Transfer learning, used in machine learning, is the reuse of a pre-trained model on a new problem. In transfer learning, a machine exploits the knowledge gained from a previous task to improve generalization about another. For example, in training a classifier to predict whether an image contains food, you could use the knowledge it gained during training to recognize drinks.

### Image data generator Keras
[article gfg](https://www.geeksforgeeks.org/cnn-image-data-pre-processing-with-generators/#:~:text=Keras%20has%20a%20module%20with%20image-processing%20helping%20tools,,batches%20of%20preprocessed%20tensors.%20Code:%20Practical%20Implementation%20:)

### Differnece between Batch Size and epochs 
-   Batch size: The batch size is the number of samples processed before updating the model. The number of epochs represents the total number of passes through the **training dataset.**
-   Epoch: It indicates the number of passes of the entire training dataset the  **machine learning**  algorithm has completed
- [article](https://medium.com/mlearning-ai/difference-between-the-batch-size-and-epoch-in-neural-network-2d2cb2a16734#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ0ZTA2Y2ViMjJiMDFiZTU2YzIxM2M5ODU0MGFiNTYzYmZmNWE1OGMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MzgyMTI1MTgsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNjY4MDEwODAzMTkzODE1Mzg2NSIsImVtYWlsIjoiZGFyc2hndXB0YTQ5MkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IkRhcnNoIEd1cHRhIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnpfU1IwSTI4eVBaaG1MQnFzaWY2X2tMcXZkdnl4ODNBSm5TeVZHPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6IkRhcnNoIiwiZmFtaWx5X25hbWUiOiJHdXB0YSIsImlhdCI6MTYzODIxMjgxOCwiZXhwIjoxNjM4MjE2NDE4LCJqdGkiOiJiNDMxZmViMzFiZjE2ZjE5YTkwMmVhMmRlNzVhNjc2MTA1NTA4ZGVjIn0.Uf2j3gQBpi1Bq_R3Fv1mJ-wXvimvOGj7EqHk73E2_gCLqpuTmqa9-AeSq7emiL4aEnG9hSpIBYohst78weDYc_MQLdd_WWqapk-C6pfGk6Mfc9Tna58TUditgDCgyNUAE1fFfICgpOov73Uje0uGlDbTZI-8SGbD0B-lre7k0GJq7bWeSVQsKnhfTW1bT420kurmSf9XWq5ImYWm9EERPWBRncRGvsg_hl0FdmG4zE6JlgfcStEAcyse0prF5tD2lIgGffckqb8tODK01dPgXGxXRXPkbwhhv9QT3RLcBMi_IS8TmJBZRp207zQDd5Q95kQ5glZbEqd-MwcfRk2Mjg)

