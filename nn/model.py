from abc import ABCMeta, abstractmethod


class Model(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass