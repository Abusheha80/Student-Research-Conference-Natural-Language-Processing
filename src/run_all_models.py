import subprocess
import os

# Define the order of scripts
script_order = [
    "dbert.py",
    "gru.py",
    "bilstm.py",
    "logistic.py",
    "randomforest.py",
    "naive.py",
    "cnn.py",
    "rnn.py",
    "lstm.py",
]

# Path to your models folder
models_folder = os.path.join(os.getcwd(), "models")

# Run each script in order
for script in script_order:
    script_path = os.path.join(models_folder, script)
    print(f"\n🟡 Running {script}...")
    try:
        result = subprocess.run(["python", script_path], check=True)
        print(f"✅ Finished {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script}: {e}")
