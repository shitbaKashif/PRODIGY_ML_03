# PRODIGY_ML_03

## Image Classification of Cats and Dogs using SVM

***Introduction***

SVM is a powerful supervised learning algorithm used for classification tasks. In this project, we leverage a pre-trained VGG16 model to extract features from images of cats and dogs, which are then used to train an SVM for classification. This approach improves accuracy by utilizing the rich feature representations learned by VGG16 from the ImageNet dataset.

***Dataset***

The dataset used in this project consists of images of cats and dogs. Ensure the dataset is organized in two directories:
  - *./train/cats:* Contains images of cats.
  - *./train/dogs:* Contains images of dogs.

***Installation***

To run this project, you need to have Python installed along with the following libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow
  - opencv-python
You can install the required libraries using pip:
  - !pip install numpy pandas scikit-learn matplotlib seaborn tensorflow opencv-python

***Usage***
  - Clone the repository or download the dataset from kaggle (https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?select=train).
  - Create a Jupiter Notebook.
  - Run the notebook to perform SVM classification and visualize the results:
      - V1.ipynb/V2.ipynb

***Results***

The script will output the following:
  - Accuracy of the SVM model on the test set.
  - Classification report showing precision, recall, and f1-score for each class.
  - Confusion matrix visualized using a heatmap.

***Further Improvements***
  - *Hyperparameter Tuning:* Use grid search or random search to find the best hyperparameters for the SVM model.
  - *Data Augmentation:* Apply data augmentation techniques such as rotation, flipping, and zooming to increase the variability of the training data.
  - *Experiment with Different CNNs:* Try other pre-trained models like ResNet50, InceptionV3, or MobileNet for feature extraction.
  - *Regularization:* Adjust regularization parameters to prevent overfitting.
