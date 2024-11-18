# Machine Learning

This repository contains code for applying feature selection techniques and supervised learning models to classify breast cancer cases. The project demonstrates how different feature extraction methods impact the performance of various classifiers.

Author
Name: Taskeen Fatima
Enrollment No.: 03-134211-045
Email: taskeenfatima2207@gmail.com

Project Structure
data.csv: The dataset used for this analysis, which includes features related to breast cancer diagnosis.
Code.ipynb: The main Jupyter Notebook containing all the code for data preprocessing, feature selection, model training, and evaluation.
Code Overview

The code performs the following tasks:

Data Loading and Preprocessing:
Loads the dataset and performs initial data exploration.
Encodes the target variable (diagnosis) for binary classification (M = 1 for malignant, B = 0 for benign).
Splits the data into training and testing sets and scales the features using StandardScaler.

Feature Selection Techniques:
Manual Feature Selection: Selects a subset of features based on domain knowledge.
Recursive Feature Elimination (RFE): Uses a logistic regression model to select the most important features.
SelectKBest: Selects the top k features based on the ANOVA F-value between the features and the target variable.

Model Training and Evaluation:
Trains and evaluates three classifiers: Decision Tree, AdaBoost, and Gradient Boosting Classifier.
Calculates performance metrics such as accuracy, precision, recall, and F1-score.
Displays confusion matrices for each model and feature selection method.

Results
The project shows how different feature selection techniques affect the classifiers' performance:

Decision Tree: Achieved consistent accuracy (~87%) across all feature selection methods.
AdaBoost: Reached 93% accuracy, demonstrating robust performance with all feature selection methods.
Gradient Boosting Classifier: Performed the best, achieving 95% accuracy across all feature extraction methods, making it the most reliable model for this dataset.

How to Run the Code
Clone this repository to your local machine.
Ensure you have Python installed with the necessary libraries (pandas, numpy, scikit-learn, matplotlib, and seaborn).
Run Code.ipynb in Jupyter Notebook or Google Colab.

Upload the data.csv file if required during execution.
Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn

Future Enhancements
Implement additional feature selection methods and classifiers.
Perform hyperparameter tuning for classifiers to optimize performance further.
Visualize feature importance for each model to better understand the selected features' impact.

License
This project is licensed under the MIT License.

Feel free to contribute or raise any issues in the repository!
