from pytorch_lightning.callbacks import EarlyStopping

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, *, wait_until: int = 100, **kwargs):
        """
        wait_until: don’t stop before this many epochs, regardless of the monitored metric.
        All other EarlyStopping args (monitor, patience, mode, etc.) work as usual.
        """
        super().__init__(**kwargs)
        self.wait_until = wait_until

    def on_validation_end(self, trainer, pl_module):
        
        # once past min_epochs, behave exactly like normal EarlyStopping
        if trainer.current_epoch > self.wait_until:
            super().on_validation_end(trainer, pl_module)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # once past min_epochs, behave exactly like normal EarlyStopping
        if trainer.current_epoch > self.wait_until:
            super().on_validation_epoch_end(trainer, pl_module)
            
    def on_train_epoch_end(self, trainer, pl_module):
        
        # once past min_epochs, behave exactly like normal EarlyStopping
        if trainer.current_epoch > self.wait_until:
            super().on_train_epoch_end(trainer, pl_module)
