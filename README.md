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
### Lesson 1: `PyTorch Basics and Gradient Descent`<br/>
**Topics Covered-:**
1. Tensors
2. Tensor Operations and Gradients
3. Interoperability with NumPy
&nbsp;
### Lesson 2: `Gradient Descent and Linear Regression with PyTorch`<br/>
**Topics Covered-:**
1. Introduction to Linear Regression
2. Building a Linear regression model from scratch
    a. Training Data <br/>
    b. Loss Function <br/>
    c. Compute Gradients <br/>
    d. Adjust Weights and Biases to reduce the loss <br/>
    e. Train the Model using Gradient Descent <br/>
    f. Train for Epochs <br/>

3. Linear Regression using PyTorch Built-ins
    a. Dataset and DataLoader <br/>
    b. nn.Linear <br/>
    c. Loss Function <br/>
    d. Optimizer <br/>
    e. Train the Model <br/>

### Lesson 3: `PyTorch Basics and Gradient Descent`<br/>
**Topics Covered-:**
1. Tensors
2. Tensor Operations and Gradients
3. Interoperability with NumPy
&nbsp;