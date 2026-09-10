# Create indices for a dataset containing 1000 samples.
indices = list(range(1000))

# Create training and validation splits.
train_indices = indices[:800]
validation_indices = indices[700:]