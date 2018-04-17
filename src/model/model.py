from abc import abstractmethod, ABC
class Model(ABC):
    @abstractmethod
    def __init__(self, input_size, joints, input_type='DEPTH', is_training=True, cacheFile=None):
        self.cacheFile = cacheFile
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def build_loss(self, weight_decay, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        pass

    @property
    def model_output(self):
        return None