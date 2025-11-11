#hyperparams
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4

GAMMA = 0.99

TAU = 0.005

BUFFER_SIZE = 10000
BATCH_SIZE = 64

HIDDEN_LAYERS = [256, 256]
ACTIVATION_FN = 'ReLU' #'Tanh'

MAX_EPISODES = 500
MAX_STEPS = 1000 #predefined by env, but can be overwritten
A_UPDATE = 2 #how often to update actor network (1 = every timestep)
C_UPDATE = 1

EXPL_NOISE = 0.2
NOISE_TYPE = 'Gaussian' #'OU'

OPTIMIZER = 'Adam'
#end of hyperparams