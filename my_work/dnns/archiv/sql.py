import os
import optuna
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BestTrialInfo:
    file: str
    value: float
    number: int
    params: Dict[str, any]

def load_best_trial_from_db(db_path: str) -> BestTrialInfo:
    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name="optuna_study", storage=storage_url)
    except KeyError:
        study = optuna.load_study(study_name=None, storage=storage_url)

    best_trial = study.best_trial
    return BestTrialInfo(
        file=os.path.basename(db_path),
        value=best_trial.value,
        number=best_trial.number,
        params=best_trial.params
    )

def main(folder_path: str) -> List[BestTrialInfo]:
    db_files = [f for f in os.listdir(folder_path) if f.endswith(".db")]

    if not db_files:
        print("No .db files found in the specified folder.")
        return []

    all_best_trials = []

    for db_file in db_files:
        db_path = os.path.join(folder_path, db_file)
        try:
            result = load_best_trial_from_db(db_path)
            all_best_trials.append(result)

            print(f"📁 File: {result.file}")
            print(f"🔢 Trial #{result.number} with value: {result.value}")
            print("🧪 Best Parameters:")
            for key, value in result.params.items():
                print(f"   - {key}: {value}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Failed to read {db_file}: {e}")

    return all_best_trials


if __name__ == "__main__":
    root = os.getcwd()
    folder_path = os.path.join(root, "studies")  # ⬅️ Replace with your actual path
    main(folder_path)

print(all_best_trials)