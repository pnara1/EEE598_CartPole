## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 wrapper.py #runs seed 0, 1, 2 for cartpole -> saves a plot of rewards over time and saves actor policy
python3 eval.py #reads actor policy from logs directory, runs evaluation on seed 10 -> generates mean+std graph


.
├── wrapper.py          # Runs training for seeds 0, 1, 2
├── eval.py             # Runs evaluation on seed 10
├── networks.py         # Actor, Critic, Replay Buffer, TD3 Agent
├── config.py           # Hyperparameters and settings
├── logs/
│   ├── seed0/
│   ├── seed1/
│   └── seed2/
└── README.md
