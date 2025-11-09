# Predicting Heart Disease with a Random Forest Classifier
  <img width="297" height="178" alt="image" src="https://github.com/user-attachments/assets/7b26e4b2-9ee4-4fa8-9528-8951f32f681c" />



  
# DEBRE TABOR UNIVERSITY 
# GAFAT INSTITUITE OF TECHNOLOGY
# DEPARTMENT OF COMPUTER SCIENCE

# Group 6 Memebers
# Student  Name          ID No.
1.	Anbesaw Alemu ………………...0656
2.	Amarech Habtamu………….…..0609
3.	Hana Sebsbe ……………………...0630
4.	Liknaw Abiyu…………………....0442
5.	Tsion Fikre ……………………..…0457
6.	Wulta Mulu …………………....…0507       

## Project Overview
This project uses the **Heart Failure Prediction Dataset** to predict the presence of heart disease based on patient medical attributes.  
The model applies a **Random Forest Classifier** and analyzes the key factors influencing heart disease risk.

## Objectives
1. Load and preprocess the dataset from Kaggle.  
2. Perform exploratory data analysis (EDA) to uncover patterns.  
3. Encode categorical variables and scale numerical features.  
4. Train and evaluate a Random Forest Classifier.  
5. Visualize performance metrics and feature correlations.

## Key Results & Visualizations
- **Accuracy:** ~85–90%  
- **Most Influential Features:** `ChestPainType`, `Cholesterol`, `Age`  
- Correlation heatmap and confusion matrix included in notebook outputs.

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

