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
    a. Tensors <br/>
    b. Tensor Operations and Gradients <br/>
    c. Torch Functions <br/>
    d. Interoperability with NumPy <br/> 
### 1.2 Linear Regression & Gradient Descent <br/>
    a. Introduction to Linear Regression <br/>
    b. Training data <br/>
    c. Linear Regression Model from scratch <br/>
    d. Loss Function <br/>
    e. Compute Gradients <br/>
    f. Adjust Weights and Biases to reduce the Loss <br/>
    g. Train the Model using Gradient Descent <br/>
    h. Train Model for multiple Epochs <br/>
### 1.3 Using PyTorch Modules: nn.Linear & nn.functional <br/>
    a. Linear Regression using PyTorch Built-ins <br/>
    b. Dataset and DataLoader <br/>
    c. nn.Linear <br/>
    d. nn.functional - Loss Function <br/>
    e. Optimizer - optim.SGD (Stochastic Gradient Descent) <br/>
    f. Train the Model <br/>
&nbsp;

#### ***Note 1:*** The Notebook (pytorch_zero_to_GANs.ipynb) contains some extra learning materials
    a. Machine Learning vs. Classical Programming <br/>
    b. Exercises and Further Reading <br/>
    c. Questions for Review <br/>
    d. Feed Forward Neural Networks <br/>
&nbsp;
## Lesson 2: `Working with Images and Logistic Regression`<br/>
**Topics Covered-:**
### 2.1 Training-Validation Split on the MNIST Dataset<br/>

### 2.2 Logistic Regression, Softmax & Cross-Entropy<br/>

### 2.3 Model Training, Evaluation & Sample Predictions<br/>