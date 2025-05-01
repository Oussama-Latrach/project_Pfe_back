class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for lower is better
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            improve = current_score > (self.best_score + self.delta)
        else:
            improve = current_score < (self.best_score - self.delta)

        if improve:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop