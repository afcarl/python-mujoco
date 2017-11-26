class Config:
    MAX_QUEUE_SIZE = 100

    # Brain parameters
    LEARNING_RATE = 0.001
    EPOCHS = 10
    NUM_STATES = 4
    NUM_ACTIONS = 1
    LOSS_V = 0.5
    LOSS_ENTROPY = 0.01

    # Predictor parameters
    PREDICTION_BATCH_SIZE = 128

    # Trainer parameters
    TRAINING_MIN_BATCH_SIZE = 32
