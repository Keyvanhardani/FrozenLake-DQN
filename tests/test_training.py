"""Test if training runs without errors"""

import os
import subprocess

def test_training():
    # Dieser Test führt das Training-Skript aus und prüft, ob es ohne Fehler (Exit-Code 0) beendet wird.
    script_path = os.path.join("..", "scripts", "train_dqn_frozenlake.py")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    assert result.returncode == 0, f"Training script failed: {result.stderr}"
    
    # Optional: Überprüfen, ob das Modell gespeichert wurde.
    model_path = os.path.join("..", "models", "trained_model.zip")
    assert os.path.exists(model_path), "Trained model was not found."
