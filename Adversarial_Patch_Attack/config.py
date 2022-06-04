def default_params():
    return {
        "batch_size": 1,  # Batch size used during training
        "train_size": 2000,  # Amount of training samples used
        "test_size": 2000,  # Amount of test samples used
        "patch_size": 0.1,  # Size of patch relative to image
        "lr": 1,  # Learning rate
        "epochs": 20,  # Number of epochs
        "patch_shape": "rectangle",  # Shape of the patch
        "target": 859,  # The target label of the attack
        "probability_threshold": 0.9,
        "max_iteration": 1000
    }
