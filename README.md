# Cancer-Risk_Prediction

This project focuses on a machine learning based predictive system to estimate an 
individual's cancer risk level based on various health and lifestyle factors. The model 
analyzes multiple factors such as demographic information, health, lifestyle, genetic and other 
factors to identify different risk categories (Low, Medium and High).

Features -

● Demographics: Age, Gender (0 = Female, 1 = Male), BMI. 

● Lifestyle & Environmental (0–10 index): Smoking, Alcohol_Use, Obesity, 
Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity, 
Physical_Activity_Level, Air_Pollution, Occupational_Hazards, Calcium_Intake.

● Genetic/Medical Flags (0/1): Family_History, BRCA_Mutation, 
H_Pylori_Infection.

● Engineered Fields: Overall_Risk_Score, Risk_Level.

EDA - 

● Data Imbalance - The patient count in the medium risk level is much more than the low and then in high risk level. 

● Cancer Types By Gender - Majority of females suffer from Breast cancer, whereas manjority of males suffer from prostrate cancer.

● Factors Contributing to High Risk Level - Air pollution, smoking , alcohol use, diet_salted_processed, Occupational hazards, diet_red_meat, obesity where the main factors Contributing to high risk
level.


Data Preprocessing - 

● Dropped columns to avoid any data leakage and correlation with the target column.

● Label Encoding - High:0, Low:1, Medium:2

Model -

● Random Forest

Data-level Resampling Technique -

● SMOTE - Improved minority class recall, reducing false negatives, which is critical in cancer risk prediction.

Hyperparameter Optimization -

● Optuna Tuning

Deployment - 

● Streamlit 

● AWS - EC2


Conclusion -

After applying different models and hyperparameter optimization, the final model achieved a 
good balance between recall and accuracy. The Optuna-tuned Class-weighted Random Forest 
provided the best performance, with strong recall for High Risk Level individuals.
