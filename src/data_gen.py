import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    def __init__(self, context, target, vocab_size, sparse=True, maxlen=50, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.context = context
        self.target = target
        self.sparse = sparse
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Number of batches per epoch
        """
        return int(np.floor(len(self.context) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        idx = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        _context = [self.context[i] for i in idx]
        y = np.array([self.target[i] for i in idx])
        X = np.array(pad_sequences(_context, maxlen=self.maxlen-1, padding='post'))
        
        if not self.sparse:
            y = to_categorical(y, num_classes=self.vocab_size)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.context))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

