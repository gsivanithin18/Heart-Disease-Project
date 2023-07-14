# Heart Disease Predictor
This project is a heart disease predictor that uses machine learning algorithms to analyze medical data and predict the likelihood of a person having heart disease. It is implemented in Python and utilizes popular libraries such as scikit-learn, pandas, and numpy. Web app is built with HTML, CSS, and Flask.

## Table of Contents
* Background
* Installation
* Usage
* Dataset
* Model Training
* Evaluation
* Contributing

## Background
Heart disease is a significant health issue worldwide, and early detection plays a crucial role in preventing complications. This predictor aims to provide a convenient tool for healthcare professionals to assess the risk of heart disease in patients based on their medical information.

The project utilizes a machine learning approach to train a predictive model using a labelled dataset of medical records. By feeding new patient data into the trained model, the predictor can estimate the probability of heart disease.

## Installation
To use this predictor, follow these steps:

1. Clone the repository: git clone https://github.com/gsivanithin18/heart-disease-predictor.git
2. Change into the project directory: cd heart-disease-predictor
3. Install the required dependencies: pip install -r requirements.txt


## Usage
To run the heart disease predictor, execute the following command:

python app.py

The program will prompt you to enter the patient's medical information, such as age, gender, blood pressure, cholesterol levels, etc. After providing the necessary details, the predictor will output the estimated probability of heart disease.


https://github.com/gsivanithin18/Heart-Disease-Project/assets/108757821/0334554c-fd85-41f3-aea7-420abb0c78c5


## Dataset
The predictor was trained using a publicly available dataset called "Heart Disease UCI" from the UCI Machine Learning Repository. The dataset contains 14 attributes, including age, sex, cholesterol levels, blood pressure, and the presence of heart disease. You can find more information about the dataset [here](https://archive.ics.uci.edu/ml/datasets/heart+disease).

Please note that the dataset used is for demonstration purposes only, and the predictor's performance may vary based on the data quality and applicability to real-world scenarios.

## Model Training
The machine learning model used for this predictor is a binary classification model. It was trained using the labelled dataset mentioned above. The training code can be found in the prediction.py file.

The prediction.py script performs the following steps:

Load and preprocess the dataset.
Split the dataset into training and testing sets.
Train a machine learning model using the training data.
Save the trained model to disk for future use.

## EDA & Evaluation
The accuracy of the predictor can be evaluated by comparing its predictions to the actual presence of heart disease in the test dataset. 
The program is trained with different classification algorithms and calculated various evaluation metrics, such as accuracy, precision, recall, and F1-score. The code can be found in [end-to-end-heart-disease-classification.ipynb](https://github.com/gsivanithin18/Heart-Disease-Project/blob/main/end-to-end-heart-disease-classification.ipynb).

## Contributing
Contributions to this heart disease predictor project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

