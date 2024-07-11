# Assignment 2
In this assignment, you need to classify images into 2 classes using simple Neural Network.

Download the [Images](./homer_bart.zip) file.

Suggestions:  
1. The images are of different dimensions, while importing/preprocessing convert all into (`64x64`) images.
2. Distribute them into batches (e.g. `batch_size = 32`).
3. Since the amount of images is less, no need to create validation dataset, go with training and test dataset only. Prefer 9:1 split between training and test dataset.
4. Use `Dense` i.e. Fully-Connecetd Layers with `activation = 'relu'` in each layer. Use `activation = 'sigmoid'` in the last layer.
5. `metrics = 'accuracy'` of test dataset will be checked only.

Mandatory:
1. Don't use `Conv2D`, `MaxPool2D` layers.
2. For passing the assignment, test dataset accuracy should be more than 90%. [Test Accuracy > 0.9]

**BEST OF LUCK**
