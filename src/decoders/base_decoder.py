from abc import ABC, abstractmethod


class BaseDecoder(ABC):

    @abstractmethod
    def decode(self):
        pass
