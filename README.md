# attention-unet
copied repo and customized from [Attention Unet](https://github.com/ozan-oktay/Attention-Gated-Networks)


# Attention-Gated Networks Setup Guide for Google Colab

This guide will walk you through setting up the environment and running the **Attention-Gated-Networks** project on **Google Colab**.

## Prerequisites

Before starting, ensure you have a **Google Colab** environment ready.

### Step 1: Clone the Repository

Start by cloning the Attention-Gated-Networks repository from GitHub:

```bash
!git clone https://github.com/ozan-oktay/Attention-Gated-Networks.git
```

### Step 2: Update and Install Python 3.9

Ensure that Python 3.9 is installed and set as the default version:

```bash
!sudo apt-get update
!sudo apt-get install python3.9
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
!sudo update-alternatives --config python3
```

### Step 3: Install `pip` and Required Python Packages

Install `pip` for Python 3.9 and the required Python packages:

```bash
!sudo apt install python3-pip
!sudo apt install python3.9-distutils
```

Verify the Python and pip versions:

```bash
!python --version
!pip --version
```

### Step 4: Change Directory

Navigate into the cloned repository:

```bash
%cd Attention-Gated-Networks/
```

### Step 5: Install Dependencies

Install the dependencies using `pip`:

```bash
!pip install git+https://github.com/ozan-oktay/torchsample.git
!pip install .
!pip install opencv-python scikit-learn
```

### Step 6: Train the Model

Navigate to the directory containing your training data and run the training script with your configuration file:

```bash
!cd '/content/drive/MyDrive/Deep Learning Lab/corrosion detection/Corrosion Condition State Classification/original/Train/train/images'
!python train_segmentation.py --config configs/config_unet_ct_multi_att_dsv_corrosion.json
```

---

Feel free to modify the training script or configuration file as needed for your use case.
