# Predicting Heart Disease with a Random Forest Classifier
  <img width="297" height="178" alt="image" src="https://github.com/user-attachments/assets/7b26e4b2-9ee4-4fa8-9528-8951f32f681c" />



  
# DEBRE TABOR UNIVERSITY 
# GAFAT INSTITUITE OF TECHNOLOGY
# DEPARTMENT OF COMPUTER SCIENCE

# Group 6 Memebers
# Student  Name          ID No.
1.	Anbesaw Alemu ………………...0656
2.	Amarech Habtamu………….…..0609
3.	Hana Sebsbe ……………………....0630
4.	Liknaw Abiyu………………….....0442
5.	Tsion Fikre ………………….…..…0457
6.	Wulta Mulu ………………….....…0507       

## Project Overview
### Heart Disease Prediction using Random Forest

A machine learning project that predicts the likelihood of heart disease using patient health metrics.
The model is built using a Random Forest Classifier and implemented in Python inside a Google Colab notebook.

## Project Features
Install dependencies 
Load dataset directly in Colab

Clean and preprocess raw data

Convert categorical features (e.g., ChestPainType) into numeric form

Train a Random Forest model

Evaluate using accuracy, classification report, and confusion matrix

Ready-to-use notebook for medical prediction research

## Dataset

Upload the dataset manually in Colab using:

from google.colab import files
files.upload()


### The dataset must include standard heart disease features such as:

Age

Sex

ChestPainType

Resting Blood Pressure

Cholesterol

Fasting Blood Sugar

Resting ECG

Max Heart Rate

Exercise-Induced Angina

Oldpeak

ST_Slope

Target (0 = No disease, 1 = Disease)

## Data Preprocessing

Main preprocessing steps:

Checking dataset structure

Handling missing values

Removing duplicates

Label encoding categorical features

Converting ChestPainType into numeric

Splitting dataset into train/test sets

## Model Used: Random Forest
 
 ## Model Evaluation

The notebook outputs:

Accuracy Score
Classification Report
Confusion Matrix

## How to Run

Open Google Colab

Upload the notebook:

Predicting_Heart_Disease_RF.ipynb


Upload your dataset when prompted

Run all cells sequentially

## Output

Cleaned dataset preview

Encoded & processed dataset

Accuracy score

Precision, Recall, F1-score

Confusion matrix

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)  
- Google Colab  
- Kaggle Dataset
  
## Conclusion:
- The Random Forest model achieved good predictive accuracy on this dataset.
- Important factors influencing heart disease likely include Age, Chest Pain Type, and Cholesterol. 
- As we observed from the output and when we compared Age, Chest Pain Type, and Cholesterol to predict heart disease.
- Cholesterol is the most influential factor from Age and Chest Pain Type, or we can say that 
  Age < Chest Pain Type < Cholesterol based on most infuantial factor in predicting heart disease in our model.
- Model can be improved by performing hyperparameter tuning (GridSearchCV) or testing other models.

