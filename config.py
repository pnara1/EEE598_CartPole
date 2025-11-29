#hyperparams
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

GAMMA = 0.99
REWARD_SCALE = 1.0

TAU = 0.01

BUFFER_SIZE = 100000
BATCH_SIZE = 64
START_TRAINING_AFTER = 2000 #max(1000, BATCH_SIZE*4)    

HIDDEN_LAYERS = [64, 64] #[256, 256] #maybe [64, 64] #[128, 128]
ACTIVATION_FN = 'ReLU' #'Tanh'

MAX_EPISODES = 500
MAX_STEPS = 1000 #predefined by env, but can be overwritten
A_UPDATE = 1 #how often to update actor network (1 = every timestep)
C_UPDATE = 1

EXPL_NOISE = 0.1
NOISE_TYPE = 'Gaussian' #'OU'

POLICY_NOISE = 0.1
NOISE_CLIP = 0.25

OPTIMIZER = 'Adam'
#end of hyperparams