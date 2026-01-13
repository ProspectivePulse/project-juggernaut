import os
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the API
app = FastAPI(title="RL Robot Brain API")

# Define the input data format
class RobotState(BaseModel):
    distance: float

@app.get("/")
def health_check():
    """Cloud Run pings this to check if the container is alive."""
    return {"status": "active", "service": "rl-robot-brain"}

@app.post("/predict")
def predict_action(state: RobotState):
    """
    Inference Endpoint.
    1. Receives sensor data.
    2. Feeds it to the loaded model.
    3. Returns the optimal action.
    """
    # Debug: Print what we received to the cloud logs
    print(f"Received state: {state}")

    # Load actual DQN model here - TBC
    dummy_action = 1

    return {"action": dummy_action}
