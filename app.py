from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load('osteoporosis_risk_xgboost_model.pkl')

# Define the input data schema using Pydantic
class PatientData(BaseModel):
    Age: int  # Changed to int to match the data type in the image
    Gender: str  # Changed to str to match the data type in the image
    Hormonal_Changes: str # Added
    Family_History: str  # Changed to str to match the data type in the image
    Race_Ethnicity: str  # Changed to str to match the data type in the image
    Body_Weight: str  # Changed to str to match the data type in the image
    Calcium_Intake: str  # Changed to str to match the data type in the image
    Vitamin_D_Intake: str  # Changed to str to match the data type in the image
    Physical_Activity: str  # Changed to str to match the data type in the image
    Smoking: str  # Changed to str to match the data type in the image
    Alcohol_Consumption: str  # Changed to str to match the data type in the image
    Medical_Conditions: str  # Changed to str to match the data type in the image
    Medications: str  # Changed to str to match the data type in the image
    Prior_Fractures: str  # Changed to str to match the data type in the image
    # No Bone_Density as it's not in the image data
    Id: int  # Added ID field


# Initialize the FastAPI app
app = FastAPI()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Osteoporosis Risk Prediction API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict_osteoporosis_risk(patient_data: PatientData):
    try:
        # Input data processing and one-hot encoding
        input_data_list = [
            patient_data.Age,
            patient_data.Gender == "Male",  # Convert to boolean (0 or 1)
            patient_data.Hormonal_Changes == "Normal",
            patient_data.Family_History == "Yes",
            patient_data.Race_Ethnicity == "Asian", # Example. Add other races as needed.
            patient_data.Body_Weight == "Underweight", # Example. Add other weights as needed.
            patient_data.Calcium_Intake == "Low", # Example. Add other intake as needed.
            patient_data.Vitamin_D_Intake == "Sufficient", # Example. Add other intake as needed.
            patient_data.Physical_Activity == "Sedentary", # Example. Add other activities as needed.
            patient_data.Smoking == "Yes",
            patient_data.Alcohol_Consumption == "Moderate", # Example. Add other consumption as needed.
            patient_data.Medical_Conditions == "Rheumatoid Arthritis", # Example. Add other conditions.
            patient_data.Medications == "Corticosteroids", # Example. Add other medications.
            patient_data.Prior_Fractures == "Yes"
        ]

        input_data = np.array([input_data_list], dtype=object) # dtype=object for mixed types

        # Convert boolean values to integers (0 or 1)
        input_data = input_data.astype(int)

        print("Input Data Shape:", input_data.shape)  # Debugging print
        print("Input Data:", input_data)  # Debugging print

        # Make a prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Return the prediction result
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))