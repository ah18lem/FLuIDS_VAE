
NUM_ROUNDS=30
NUM_CLIENTS=10

FRACTION_FIT=1  # Sample 10% of available clients for training
FRACTION_EVALUATE=1  # Sample 5% of available clients for evaluation



MIN_FIT_CLIENTS=10 # Never sample less than 10 clients for training
MIN_EVALUATE_CLIENTS=10# Never sample less than 5 clients for evaluation
MIN_AVAILABLE_CLIENTS=10


VAE=False


EPOCHS_CLIENT=5
EPOCHS_SERVEUR=20


BATCH_SIZE=64
LEARNING_RATE=0.0001

RATIO_LABEL=0.3


MULTICLASS=True
INPUT_DIM=45

NUM_CLASSES=6
  
TRAINING_DATA="train_regroupe.csv"
TESTING_DATA="test_regroupe.csv"


MULTICLASS_TARGET_COL="label_grouped"

