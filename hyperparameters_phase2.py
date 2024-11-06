# Random Seed
SEED = 42


# Input Config

NUM_SPECTROGRAM_BINS = 513

NUM_MEL_BINS = 128

LOWER_EDGE_HERTZ = 80.0

UPPER_EDGE_HERTZ = 7600.0

SAMPLE_RATE = 16000


FRAME_LENGTH = 1024
FRAME_STEP = 256
FFT_LENGTH=1024

N_MFCC = 40



# Training Config
EPOCHS = 100

BATCH_SIZE = 32

K_FOLD = 10


# Regularizer Hyperparameter
L2 = 1e-6
DROPOUT = 0.3


# Base Directory for Datasets
BASE_DIRECTORY = "data"


# Optimizer Config
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY_PARAMETERS = -0.15
LEARNING_RATE_DECAY_STRATPOINT = 50
LEARNING_RATE_DECAY_STEP = 20



# Loss Config
GAMMA = 2


# warmup


# # Random Seed
# SEED = 42

# # Input Config
# NUM_SPECTROGRAM_BINS = 513
# NUM_MEL_BINS = 128
# LOWER_EDGE_HERTZ = 80.0
# UPPER_EDGE_HERTZ = 7600.0
# SAMPLE_RATE = 16000

# FRAME_LENGTH = 1024
# FRAME_STEP = 256
# FFT_LENGTH = 1024
# N_MFCC = 40

# # Training Config
# EPOCHS = 100
# BATCH_SIZE = 32
# K_FOLD = 10

# # Regularizer Hyperparameter
# L2 = 1e-6
# DROPOUT = 0.3

# # Base Directory for Datasets
# BASE_DIRECTORY = "data"

# # Optimizer Config (Updated for Warm-Up and Cosine Annealing)
# LEARNING_RATE = 1e-4
# LEARNING_RATE_DECAY_PARAMETERS = -0.15
# LEARNING_RATE_DECAY_STRATPOINT = 50
# LEARNING_RATE_DECAY_STEP = 20



# # Define the learning rate parameters
# MAX_LEARNING_RATE = 0.001  # Set your desired maximum learning rate here
# MIN_LEARNING_RATE = 1e-6   # Set your desired minimum learning rate
# WARMUP_EPOCHS = 5          # Number of epochs for the warm-up phase
# TOTAL_EPOCHS = EPOCHS          # Total number of training epochs

# # Loss Config
# GAMMA = 2
