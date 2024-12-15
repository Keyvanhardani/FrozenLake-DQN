"""Test the evaluation logic"""

import os
import subprocess

def test_evaluation():
    
    model_path = os.path.join("..", "models", "trained_model.zip")
    assert os.path.exists(model_path), "No trained model found for evaluation."
    
    # Ein echter Evaluations-Test würde hier z. B. ein separates evaluation.py Skript starten:
    # result = subprocess.run(["python", "../scripts/evaluate_model.py"], capture_output=True, text=True)
    # assert result.returncode == 0, f"Evaluation script failed: {result.stderr}"
