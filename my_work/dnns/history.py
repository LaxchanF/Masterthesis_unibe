import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt

# Use a raw string (r"...") for Windows paths, or double backslashes
study_path = r"sqlite:///C:\Masterthesis_unibe\my_work\dnns\studies\diverse\convnext_diverse_study.db"

# Load the study
study = optuna.load_study(
    study_name="convnext_diverse_study",  # Update if your actual name is different
    storage=study_path
)

# --- Plot 1: Optimization history ---
fig1 = vis.plot_optimization_history(study)
fig1.show()

# --- Plot 2: Parameter importance ---
fig2 = vis.plot_param_importances(study)
fig2.show()

# --- Plot 3: Convergence / Best-so-far values ---
best_values = []
current_best = float('inf')
for t in study.trials:
    if t.value is not None and t.value < current_best:
        current_best = t.value
    best_values.append(current_best)

plt.figure()
plt.plot(best_values)
plt.xlabel("Trial")
plt.ylabel("Best Objective Value So Far")
plt.title("Convergence Plot")
plt.grid(True)
plt.show()
