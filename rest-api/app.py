import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List, Dict, Any

# Create a FastAPI application instance
app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow_server:5000")

def test_model_exists(model: Any, model_uri: str) -> None:
    """
    Checks if a model exists.

    Parameters:
    - model (Any): The model to check.
    - model_uri (str): The URI of the model.

    Raises:
    - HTTPException: If the model is None, indicating that the model does not exist, 
      raises an HTTPException with status code 500 and a message indicating the model URI.
      
    Returns:
    - None
    """
    if model is None:
        # If model is None, raise an HTTPException with status code 500
        # indicating that the model does not exist
        raise HTTPException(status_code=500, detail=f'Model "{model_uri}" not found in MLFlow!')


def load_model(model_uri: str) -> Union[mlflow.pyfunc.PythonModel, None]:
    """
    Loads an MLflow model.

    Parameters:
    - model_uri (str): The URI of the MLflow model.

    Returns:
    - Union[mlflow.pyfunc.PythonModel, None]: The loaded MLflow model if successful,
      otherwise returns None.
    """
    try:
        # Attempt to load the MLflow model
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        # If loading the model fails, set model to None
        model = None 

    return model

# Define URI for the diabetes model
DIABETES_MODEL_URI = "models:/diabetes_model@staging"

# Load the diabetes model
diabetes_model = load_model(DIABETES_MODEL_URI)

# Define Pydantic model for input data to the diabetes model
class DiabetesDataModel(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

# Endpoint for making predictions with the diabetes model
@app.post("/diabetes/predict")
async def diabetes_model_predict(data_list: List[DiabetesDataModel]) -> Dict:
    # Check if diabetes_model exists, raise HTTPException if not
    test_model_exists(diabetes_model, DIABETES_MODEL_URI)
    
    # Convert input data to DataFrame
    df = pd.DataFrame([row.dict() for row in data_list])
    
    # Make predictions using the diabetes model
    predictions =  diabetes_model.predict(df)

    # Return predictions
    return {
        "predictions": predictions.tolist()
    }

# Endpoint for getting metadata of the diabetes model
@app.get("/diabetes/metadata")
async def diabetes_model_metadata() -> Dict:
    # Check if diabetes_model exists, raise HTTPException if not
    test_model_exists(diabetes_model, DIABETES_MODEL_URI)

    # Return metadata of the diabetes model
    return {
        "metadata": diabetes_model.metadata.to_dict()
    }

# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
