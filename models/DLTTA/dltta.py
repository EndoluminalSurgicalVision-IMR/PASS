import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import copy
from numpy.linalg import norm


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size):

        self.memory = {}
        self.size = size

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        dimension = 131072
        avg = []

        for i, key in enumerate(keys):

            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])

            self.memory.update(
                {key.reshape(dimension).tobytes(): (logits[i])})

    def _prepare_batch(self, sample, attention_weight):

        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        print(attention_weight)
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []

        dimension = 131072
        keys = keys.reshape(len(keys), dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples)



def configure_model(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
    return model
