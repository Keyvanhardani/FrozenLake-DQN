# Project: FrozenLake DQN Trainer

This project provides a training pipeline for a Deep Q-Network (DQN) agent on the `FrozenLake-v1` environment (from Gymnasium). It demonstrates how to set up, train, and evaluate a DQN agent using `stable-baselines3`. 

### What the Code Does

1. **Environment Initialization:**  
   Creates a `FrozenLake-v1` environment with a 4x4 grid. The agent must navigate from start to goal while avoiding traps. It's set to `is_slippery=False` for deterministic behavior.

2. **DQN Agent Setup:**  
   Configures DQN with specific hyperparameters (learning rate, exploration settings, etc.) and uses an MLP policy as the approximator.

3. **Training:**  
   Runs training for a specified number of timesteps. A callback periodically evaluates the model and logs performance metrics.

4. **Evaluation:**  
   After training, the model is evaluated over multiple episodes. The mean reward and standard deviation are reported.

### Test Plan

- **Environment Tests:** Check if environments load and reset properly.
- **Training Tests:** Run short training sessions to ensure no runtime errors and that models save correctly.
- **Evaluation Tests:** Confirm that evaluation runs smoothly and produces expected results.
- **Extensibility:** Try different environments (e.g., `CartPole-v1`) or adjust hyperparameters to verify flexible usage.

---

Dieses Projekt bietet eine Trainingspipeline für einen Deep Q-Network (DQN)-Agenten in der `FrozenLake-v1` Umgebung. Es demonstriert die Einrichtung, das Training und die Evaluation eines DQN-Agenten mithilfe von `stable-baselines3`.

### Funktionsweise des Codes

1. **Umgebungserstellung:**  
   Es wird eine `FrozenLake-v1` Umgebung mit einem 4x4-Gitter erstellt. Der Agent muss vom Start zum Ziel navigieren, ohne in Löcher zu fallen. Mit `is_slippery=False` ist das Verhalten deterministisch.

2. **DQN-Agent-Konfiguration:**  
   Der DQN-Agent wird mit bestimmten Hyperparametern (Lernrate, Explorationsstrategien etc.) versehen und nutzt eine MLP-Policy als Approximator.

3. **Training:**  
   Das Training läuft über eine festgelegte Anzahl von Schritten. Ein Callback führt regelmäßig Evaluationen durch, um den Lernfortschritt zu überprüfen.

4. **Evaluation:**  
   Nach dem Training wird das Modell über mehrere Episoden getestet. Die durchschnittliche Belohnung und deren Standardabweichung werden ausgegeben.

### Testplan

- **Umgebungs-Tests:** Überprüfen, ob die Umgebung korrekt geladen und zurückgesetzt wird.
- **Training-Tests:** Kurzes Training durchführen, um sicherzustellen, dass keine Fehler auftreten und das Modell gespeichert wird.
- **Evaluations-Tests:** Sicherstellen, dass die Evaluation ohne Fehler läuft und sinnvolle Ergebnisse liefert.
- **Erweiterbarkeit:** Andere Umgebungen ausprobieren (z.B. `CartPole-v1`) oder Hyperparameter anpassen, um die Flexibilität zu testen.
