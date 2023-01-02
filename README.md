# Deep Learning with PyTorch: Zero-To-GANs

## Software And Tools Requirements

1. [GithubAccount](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/downloads)
5. [AnacondaPackage/JupyterNoteBook](https://www.anaconda.com/products/distribution)

## Creata a New Environment and Activate!!

```
conda create -p torchvenv python==3.9 -y
conda activate torchvenv/
```

## Install all the Required Libraries!!

```
pip3 install -r requirements.txt
```

## Run the below Command to Install PyTorch

*`Note - If you are using Jupyter then follow the below steps to Enbale GPU in your Local Environment before executing the pip-command`*<br/>

* Step 1: Download and Install a supported version of Microsoft Visual Studio using this [link](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)
* Step 2: Download and Install CUDA Toolkit using this [link](https://developer.nvidia.com/cuda-downloads?)


**With GPU**: `for the devices with CUDA enabled GPU`

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

**With CPU**: `for devices with CPU only`

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
.
.
.
.
## Getting started with the Codes!!
&nbsp;
## Lesson 1: `PyTorch Basics and Gradient Descent`<br/>
**Topics Covered-:**
### 1.1 PyTorch basics: Tensors, Gradients, and Autograd <br/>
    a. Tensors 
    b. Tensor Operations and Gradients 
    c. Torch Functions 
    d. Interoperability with NumPy  
### 1.2 Linear Regression & Gradient Descent <br/>
    a. Introduction to Linear Regression 
    b. Training data
    c. Linear Regression Model from scratch 
    d. Loss Function 
    e. Compute Gradients
    f. Adjust Weights and Biases to reduce the Loss 
    g. Train the Model using Gradient Descent
    h. Train Model for multiple Epochs
### 1.3 Using PyTorch Modules: nn.Linear & nn.functional <br/>
    a. Linear Regression using PyTorch Built-ins 
    b. Dataset and DataLoader
    c. nn.Linear 
    d. nn.functional - Loss Function 
    e. Optimizer - optim.SGD (Stochastic Gradient Descent)
    f. Train the Model
&nbsp;

#### ***Note 1:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Machine Learning vs. Classical Programming 
    b. Exercises and Further Reading 
    c. Questions for Review 
    d. Feed Forward Neural Networks 
&nbsp;

## Lesson 2: `Working with Images and Logistic Regression`<br/>
**Topics Covered-:**
### 2.1 Training-Validation Split on the MNIST Dataset<br/>
    a. Setting up the Datasets: Working with Images
    b. Training and Validation Datasets
### 2.2 Logistic Regression, Softmax & Cross-Entropy<br/>
    a. Model 
    b. Softmax Function 
    c. Evaluation Metric and Loss Function 
    d. Cross-Entropy 
### 2.3 Model Training, Evaluation & Sample Predictions<br/>
    a. Training the model 
    b. Validating the model 
    c. Testing with individual images 
    d. Sample Predictions 
    e. Saving and loading the model
&nbsp;

#### ***Note 2:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Exercises
    b. Summary and Further Reading
&nbsp;

## Lesson 3: `Training Deep Neural Networks on a GPU`<br/>
**Topics Covered-:**
### 3.1 Activation Functions, Non-Linearity & Backprop<br/>
    a. Preparing the Dataset 
    b. Using random_split helper Function 
    c. Hidden Layers, Activation Functions and Non-Linearity 
    d. Model 
### 3.2 Training Models faster using Cloud GPUs(Local GPU in my Case)<br/>
    a. Checking for all the Available Devices and Versions
    b. Using a GPU 
### 3.3 Experimenting with Hyperparameters to Improve the Model<br/>
    a. Training the Model 
    b. Testing with individual images
&nbsp;

#### ***Note 3:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Exercises
    b. Summary and Further Reading
&nbsp;

## Lesson 4: `Image Classification with Convolutional Neural Networks(CNN)`<br/>
**Topics Covered-:**
### 4.1 Working with 3-Channel RGB Images<br/>
    a. Exploring the CIFAR10 Dataset
    b. Training and Validation Datasets
### 4.2 Convolutions, Kernels & Features Maps<br/>
    a. Defining the Model (Convolutional Neural Network)
    b. Kernels and Feature Maps
    c. Using the GPU 
    d. Training the Model
### 4.3 Training Curve, Underfitting & Overfitting<br/>
    a. Testing with individual images
    b. Saving and loading the model
&nbsp;

#### ***Note 4:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Summary and Further Reading/Exercises
&nbsp;

## Lesson 5: `Data Augmentation, Regularization and ResNets`<br/>
**Topics Covered-:**
### 5.1 Adding Residual Layers with Batchnorm to CNNs<br/>
    a. Preparing the CIFAR10 Dataset
    b. Data Augmentation and Normalization
    c. Using a GPU
    d. Model with Residual Blocks and Batch Normalization
### 5.2 Learning Rate Annealing, Weight Decay & More<br/>
    a. Training the model
### 5.3 Training a State-Of-The-Art Model in 5 Minutes<br/>
    a. Using Adam Optimizer
    b. Testing with individual images
&nbsp;

#### ***Note 5:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Summary and Further Reading/Exercises
&nbsp;

## Lesson 6: `Generative Adversarial Networks and Transfer Learning`<br/>
**Topics Covered-:**
### 6.1 Generating Fake Digits & Anime Faces with GANs<br/>
    a. Introduction to Generative Modeling
    b. Downloading and Exploring the Data 
### 6.2 Training Generator and Discriminator Networks<br/>
    a. Using a GPU
    b. Discriminator Network 
    c. Generator Network 
    d. Discriminator Training
    e. Generator Training
    f. Full Training Loop 
### 6.3 Transfer Learning for Image Classification<br/>
    a. Downloading the Dataset 
    b. Creating a Custom PyTorch Dataset
    c. Creating Training and Validation Sets
    d. Modifying a Pretrained Model (ResNet34)
    e. GPU Utilities and Training Loop
    f. Finetuning the Pretrained Model
    g. Training a model from scratch 
&nbsp;
