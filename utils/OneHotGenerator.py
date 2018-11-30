import torch
import numpy as np

class OneHotGenerator():
    '''
    This class creates a-one-hot-tesnsor from an integer.
    Input: 
        num_labels - number of tensors to create
        mode - tensors can be a numpy array ('numpy') or a pytorch tensor('torch')
    '''
    def __init__(self, num_labels, mode = 'numpy'):
        self.mode = mode
        self.num_labels = num_labels
        if (mode == 'torch'):
            self.one_hot_tensor = torch.eye(self.num_labels)
        else:
            self.one_hot_tensor = np.eye(self.num_labels)

    def to_one_hot(self, label):
        '''
        Convert an integer to one-hot-tensor
        Input: int
        Output: tensor
        '''
        assert 0 <= label < self.num_labels , "OutOfRange"
        return self.one_hot_tensor[label]

    def to_number(self, one_hot_tensor):
        '''
        Convert a one-hot-tensor to an integer
        Input: one-hot-tensor
        Output: integer
        '''
        assert one_hot_tensor.shape[0] == self.num_labels, "OutOfRange"
        if (self.mode) == 'torch' and ('torch' in str(type(one_hot_tensor))):
            return np.argmax(one_hot_tensor.numpy())
        else:
            return np.argmax(one_hot_tensor)
