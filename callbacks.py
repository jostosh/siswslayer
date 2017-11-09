from tensorflow.contrib.keras.python.keras.callbacks import Callback
import time


def sec_to_hours_minutes_sec(sec):
    hours, sec = divmod(int(sec), 3600)
    minutes, sec = divmod(sec, 60)
    return "{} hours, {} minutes and {} seconds".format(hours, minutes, sec)


class EstimateTimeRemaining(Callback):

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.t0 = None
        self.epoch_nbr = 0

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        t_delta = time.time() - self.t0
        time_per_epoch = t_delta / (epoch + 1)
        time_remaining = (self.total_epochs - (epoch + 1)) * time_per_epoch
        print("\nEstimated time after {} epochs".format(epoch + 1))
        print("Time per epoch:    ", sec_to_hours_minutes_sec(time_per_epoch))
        print("Time elapsed:      ", sec_to_hours_minutes_sec(t_delta))
        print("Time remaining:    ", sec_to_hours_minutes_sec(time_remaining))
