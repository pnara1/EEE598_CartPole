import os, sys
import subprocess 


TRAINING_SEEDS = [0, 1, 2]
EVAL_SEED = 10

#call python3 cartpole.py <seed>

for seed in TRAINING_SEEDS:
    print(f"=== Training seed {seed} ===")
    subprocess.run(["python3", "cartpole.py", str(seed)])