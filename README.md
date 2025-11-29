python3 -m venv venv
source activate venv/bin/activate
python3 wrapper.py #runs seed 0, 1, 2 for cartpole -> saves a plot of rewards over time and saves actor policy
python3 eval.py #reads actor policy from logs directory, runs evaluation on seed 10 -> generates mean+std graph 
