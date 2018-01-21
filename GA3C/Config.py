class Config:
    MAX_QUEUE_SIZE = 100

    # Brain parameters
    LEARNING_RATE = 5e-3
    NUM_STATES = 4
    NUM_ACTIONS = 2
    LOSS_V = 0.5
    LOSS_ENTROPY = 0.01

    N_STEP_RETURN = 8
    GAMMA = 0.99
    GAMMA_N = GAMMA ** N_STEP_RETURN

    # Predictor parameters
    N_PREDICTORS = 1
    PREDICTION_BATCH_SIZE = 8

    # Trainer parameters
    N_TRAINERS = 1
    TRAINING_MIN_BATCH_SIZE = 32

    # Agent Parameters
    N_AGENTS = 8
    EPSILON_START = 1
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.9999
