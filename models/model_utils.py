class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        # If the new validation loss is lower, update min_validation_loss and reset counter.
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # If the new validation loss is not lower, increment the counter.
        else:
            self.counter += 1
            # If the counter reaches patience, suggest to stop.
            if self.counter >= self.patience:
                return True
        return False




