import random
import torch
import numpy as np
from collections import deque

class Storage:
    def __init__(self, storage_size=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if storage_size is None:
            self.buffer = deque()
        else:
            self.buffer = deque(maxlen=int(storage_size))

    def store(self, data):
        '''stored a single group of data'''
        def fix(x):
            if isinstance(x[0], np.bool_): x = 1 - x
            if not isinstance(x, np.ndarray): x = np.array(x.cpu())
            if len(x.shape) == 0: x = np.expand_dims(x, axis=0)
            return x

        transition = tuple(fix(x) for x in data)
        self.buffer.append(transition)

    def get(self, source):
        '''return all data from the given source'''

        # group together all data of the same type
        n = len(self.buffer[0])
        data = [torch.FloatTensor(np.array([arr[i] for arr in source])).to(self.device) for i in range(n)]

        # expend data dimensions until they all have the same number of dimensions
        max_dim = max([len(d.shape) for d in data])
        for i in range(len(data)):
            while len(data[i].shape) < max_dim:
                data[i].unsqueeze_(2)
        return data

    def get_all(self):
        '''return all stored data'''
        return self.get(self.buffer)

    def sample(self, batch_size=1):
        '''return a random sample from the stored data'''
        batch_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, batch_size)
        return self.get(batch)

    def get_batches(self):
        num_batches = max(1, len(self.buffer) // batch_size)
        for _ in range(num_batches):
            yield self.sample()

    def clear(self):
        '''clear stored data'''
        self.buffer.clear()
