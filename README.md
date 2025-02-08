Osteoporosis Risk Prediction

Osteoporosis is a bone disease that leads to an increased risk of fractures due to decreased bone density. It is particularly prevalent among older adults and postmenopausal women.Early prediction of osteoporosis risk can help in preventive care and reduce healthcare costs.

This project aims to predict the risk of osteoporosis in patients based on their health data using machine learning. The project follows the entire machine learning lifecycle, from data exploration and preprocessing to model training, evaluation, and deployment.

Project Overview
- Objective: Predict the risk of osteoporosis in patients using health data.  
- Target Variable: Binary classification (1 = At Risk, 0 = Not At Risk).  
- Model: XGBoost (Extreme Gradient Boosting).  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
---
 Dataset
The dataset used in this project is osteoporosis.csv. It contains 1958 rows (patients) and 16 columns (features). Below is a description of the features:

| Feature           | Description                                      |
|------------------------|------------------------------------------------------|
| Id                     | Unique identifier for each patient                   |
| Age                    | Age of the patient (in years)                        |
| Gender                 | Gender of the patient (Male/Female)                  |
| Hormonal Changes       | Hormonal status (e.g., Normal, Postmenopausal)       |
| Family History         | Family history of osteoporosis (Yes/No)              |
| Race/Ethnicity         | Race or ethnicity of the patient                     |
| Body Weight            | Body weight category (e.g., Underweight, Normal)     |
| Calcium Intake         | Daily calcium intake (Low/Adequate)                  |
| Vitamin D Intake       | Vitamin D intake level (Insufficient/Sufficient)     |
| Physical Activity      | Level of physical activity (Sedentary/Active)        |
| Smoking Status         | Smoking status of the patient (Smoker/Non-smoker)    |
| Alcohol Consumption    | Alcohol consumption level (None/Moderate/Heavy)      |
| Medical Conditions     | Existing medical conditions (e.g., Rheumatoid Arthritis) |
| Medications            | Medications taken (e.g., Corticosteroids)            |
| Prior Fractures        | History of prior fractures (Yes/No)                  |
| Osteoporosis           | Target variable (1 = At Risk, 0 = Not At Risk)       |


 Installation
1. Clone the repository:
      git clone https://github.com/Hanah29/machin-learning-project.git
       cd machin-learning-project
   
2. Install the required dependencies:
      pip install -r requirements.txt
   
 Usage
1. Data Exploration and Preprocessing:
   - Run the Jupyter Notebook data_exploration.ipynb to explore the dataset and perform preprocessing steps.

2. Model Training and Evaluation:
   - Run the Python script train_model.py to train the XGBoost model and evaluate its performance.
          python train_model.py
     
3. Model Deployment (Optional):
   - Deploy the trained model as an API using FastAPI:
          uvicorn app:app --reload
     
   - Test the API using Postman or cURL:
          curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Age": 65, "BMI": 25, ...}'
     ---
Results
- Model Performance:
  - Accuracy: 0.8622  
  - Precision: 0.9277  
  - Recall: 0.7857  
  - F1-Score: 0.8508  
  - ROC-AUC: 0.8910  

- Comparison Against Baseline:
  - Baseline Accuracy: 0.5000 (Dummy Classifier)  
  - XGBoost Accuracy: 0.8622  

Deployment
The trained model can be deployed as an API using FastAPI. Follow these steps to deploy and test the API:

1. Run the FastAPI server:
      uvicorn app:app --reload
   

2. Test the API using Postman or cURL:
      curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Age": 65, "BMI": 25, ...}'
  
 Author
Hana Ayalew Mhiret 
 id -DBU1401373
 
 License
This project is licensed under the MIT License. 



