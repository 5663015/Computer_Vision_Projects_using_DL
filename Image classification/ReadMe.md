# Introduction

This image classification project tries to solve 'Cat V.S. Dog' problem in Kaggel. It's a binary classification problem, we need to correctly identify whether a picture is a cat or a dog. 
# Dataset

[Dataset download](https://www.kaggle.com/c/dogs-vs-cats/data)

The directory of the original dataset is as follows:
- test1.zip
  - 1.jpg
  - 2.jpg
  - ...
- train.zip
  - cat.0.jpg
  - cat.1.jpg
  - ...
  - dog.0.jpg
  - dog.1.jpg
  - ...

Ther are no labels in test.zip, so we won't use test.zip. The train.zip file contains 25000 images of cats and dogs. There are 12,500 pictures of cats and dogs respectively. You need to divide them into training set, test set, and validation set, and put them in 'data' directory:

- data
  - train
  - test
  - val

# Implement

Here we fine-tuned on a pre-trained Inception-V3 model. The code is implemented using tensorflow 1.10 and tf.keras core module.
