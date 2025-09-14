# CNN-model-Dog-vs-Cat
🐶🐱 A Convolutional Neural Network (CNN) model to classify images of cats and dogs using TensorFlow & Keras.

# 🐶🐱 Cats vs Dogs Classification with CNN  

**CODE_BY:** [tech_sarwesh](https://github.com/tech-sarwesh)  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)  

---

## 📖 Project Description  
This project implements a **Convolutional Neural Network (CNN)** to classify images of **cats and dogs** using **TensorFlow/Keras**.  

The workflow includes:  
✔️ Image preprocessing (resizing, normalization, train/validation split)  
✔️ Building & training a CNN model from scratch  
✔️ Evaluating model accuracy & loss on unseen test data  
✔️ Visualizing confusion matrix for performance analysis  
✔️ Predicting on custom images with visualization  

---

## 📦 Requirements  

```bash
numpy
matplotlib
seaborn
opencv-python
pillow
tensorflow
scikit-learn
```

## ⚙️ Installation

Follow these steps to set up the project:
```bash
git clone git@github.com:Tech-sarwesh/CNN-model-Dog-vs-Cat.git
cd CNN-model-Dog-vs-Cat
pip install -r requirements.txt
```
---
## Update(SOON.....)

In Update - 
- Data Augmentation
- Add API feature

## 📂 Project Structure

- 📁 CNN-model-Dog-vs-Cat/
- 📁 dog_cat
  - 📁 catvsdogs/
    - 📁 train/
      - 🐱 cats/
      - 🐶 dogs/
    - 📁 test/
      - 🐱 cats/
      - 🐶 dogs/
  - 📁 test
    - 🐱 cats/
    - 🐶 dogs/
  - 📁 train
    - 🐱 cats/
    - 🐶 dogs/
- 📁 notebooks/
  - 📄 data_visualization.ipynb
- 📁 src/
  - 📄 data_loader.py
  - 📄 model.py
  - 📄 train.py
  - 📄 evaluate.py
  - 📄 predict.py
- 📄 requirements.txt
- 📄 README.md


