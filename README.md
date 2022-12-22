# Deep Learning with PyTorch: Zero-To-GANs

## Software And Tools Requirements

1. [GithubAccount](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/downloads)
5. [AnacondaPackage/JupyterNoteBook](https://www.anaconda.com/products/distribution)

## Creata a New Environment and Activate!!

```
conda create -p venv python==3.9 -y
conda activate venv/
```

## Install all the Required Libraries!!

```
pip3 install -r requirements.txt
```

## Run the below Command to Install PyTorch

**With GPU**: `for the devices with CUDA enabled GPU`

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
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
### 3.1 Multilayer Neural Networks using nn.Module<br/>
### 3.2 Activation Functions, Non-Linearity & Backprop<br/>
### 3.3 Training Models faster using Cloud GPUs<br/>
&nbsp;

#### ***Note 3:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Exercises
    b. Summary and Further Reading
&nbsp;