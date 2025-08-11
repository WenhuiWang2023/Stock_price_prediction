# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
from keras.models import load_model



# Define input data schema
class PatientData(BaseModel):

    s1: float   # serum measurement 1
    s2: float   # serum measurement 2  
    s3: float   # serum measurement 3
    s4: float   # serum measurement 4
    s5: float   # serum measurement 5
    s6: float   # serum measurement 6
    s7: float   # serum measurement 1
    s8: float   # serum measurement 2  
    s9: float   # serum measurement 3
    s10: float   # serum measurement 4
    s11: float   # serum measurement 5
    s12: float   # serum measurement 6
    s13: float   # serum measurement 3
    s14: float   # serum measurement 4
    s15: float   # serum measurement 5
    s16: float   # serum measurement 6
    s17: float   # serum measurement 1
    s18: float   # serum measurement 2  
    s19: float   # serum measurement 3
    s20: float   # serum measurement 4
    s21: float   # serum measurement 1
    s22: float   # serum measurement 2  
    s23: float   # serum measurement 3
    s24: float   # serum measurement 4
    s25: float   # serum measurement 5
    s26: float   # serum measurement 6
    s27: float   # serum measurement 1
    s28: float   # serum measurement 2  
    s29: float   # serum measurement 3
    s30: float   # serum measurement 4
    s31: float   # serum measurement 5
    s32: float   # serum measurement 6
    s33: float   # serum measurement 3
    s34: float   # serum measurement 4
    s35: float   # serum measurement 5
    s36: float   # serum measurement 6
    s37: float   # serum measurement 1
    s38: float   # serum measurement 2  
    s39: float   # serum measurement 3
    s40: float   # serum measurement 4
    s41: float   # serum measurement 5
    s42: float   # serum measurement 6
    s43: float   # serum measurement 3
    s44: float   # serum measurement 4
    s45: float   # serum measurement 5
    s46: float   # serum measurement 6
    s47: float   # serum measurement 1
    s48: float   # serum measurement 2  
    s49: float   # serum measurement 3
    s50: float   # serum measurement 4
    s51: float   # serum measurement 5
    s52: float   # serum measurement 6
    s53: float   # serum measurement 3
    s54: float   # serum measurement 4
    s55: float   # serum measurement 5
    s56: float   # serum measurement 6
    s57: float   # serum measurement 1
    s58: float   # serum measurement 2  
    s59: float   # serum measurement 3
    s60: float   # serum measurement 4
    
    class Config:
        schema_extra = {
            "example": {
                "s1": 0.451,
                "s2": 0.456,
                "s3": 0.457,
                "s4": 0.469,
                "s5": 0.470,
                "s6": 0.488,
                "s7": 0.507,
                "s8": 0.515,
                "s9": 0.506,
                "s10": 0.500,
                "s11": 0.474,
                "s12": 0.500,
                "s13": 0.505,
                "s14": 0.557,
                "s15": 0.559,
                "s16": 0.554,
                "s17": 0.576,
                "s18": 0.589,
                "s19": 0.582,
                "s20": 0.576,
                "s21": 0.569,
                "s22": 0.589,
                "s23": 0.581,
                "s24": 0.593,
                "s25": 0.591,
                "s26": 0.598,
                "s27": 0.587,
                "s28": 0.588,
                "s29": 0.571,
                "s30": 0.550,
                "s31": 0.552,
                "s32": 0.553,
                "s33": 0.567,
                "s34": 0.540,
                "s35": 0.536,
                "s36": 0.532,
                "s37": 0.524,
                "s38": 0.528,
                "s39": 0.527,
                "s40": 0.521,
                "s41": 0.535,
                "s42": 0.536,
                "s43": 0.550,
                "s44": 0.564,
                "s45": 0.524,
                "s46": 0.531,
                "s47": 0.522,
                "s48": 0.510,
                "s49": 0.515,
                "s50": 0.505,
                "s51": 0.490,
                "s52": 0.484,
                "s53": 0.470,
                "s54": 0.476,
                "s55": 0.472,
                "s56": 0.487,
                "s57": 0.474,
                "s58": 0.467,
                "s59": 0.479,
                "s60": 0.500
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0"
)
 
# Load the trained model and scaler
model_path = os.path.abspath('model/LSTM_delta_AMZ.h5')
model = load_model(model_path)
scaler_path = os.path.join("model", "scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.post("/predict")
def predict_progression(patient: PatientData):
    """
    Predict diabetes progression score
    """
    # Convert input to numpy array
    features = np.array([
        patient.s1, patient.s2, patient.s3, patient.s4,patient.s5, patient.s6, patient.s7, patient.s8,patient.s9, patient.s10, patient.s11, patient.s12,patient.s13, patient.s14, patient.s15, patient.s16, patient.s17, patient.s18,patient.s19, patient.s20, patient.s21, patient.s22, patient.s23, patient.s24,patient.s25, patient.s26, patient.s27, patient.s28,patient.s29, patient.s30, patient.s31, patient.s32, patient.s33, patient.s34, patient.s35, patient.s36, patient.s37, patient.s38,patient.s39, patient.s40, patient.s41, patient.s42, patient.s43, patient.s44, patient.s45, patient.s46, patient.s47, patient.s48, patient.s49, patient.s50, patient.s51, patient.s52, patient.s53, patient.s54, patient.s55, patient.s56, patient.s57, patient.s58,patient.s59, patient.s60
    ])
    
    # Make prediction
    #prediction = model.predict(features)[0]
    input = np.reshape(features, (1, 60, 1))
    prediction = model.predict(input)
    prediction = prediction.flatten() + features[-1]
    prediction = prediction.reshape(-1, 1)
    prediction = scaler.inverse_transform(prediction)
    
    # Return result with additional context
    return {
        "predicted_progression_score": [float(np.round(x,2)) for x in prediction],
        #"interpretation": get_interpretation(prediction)
    }
 
def get_interpretation(score):
    """Provide human-readable interpretation of the score"""
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"
    


@app.get("/")
def health_check():
    return {"status": "healthy", "model": "diabetes_progression_v1"}