import random
import pickle

import torch
import torch.nn as nn

class DRQN(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 output_size):
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.initial_weights = self.checkpoint_weights()

    def forward(self, xin, hidden):
        z, new_hidden = self.rnn(xin, hidden)
        return self.output(z), new_hidden

    def sample_action(self, xin, hidden, epsilon):
        output = self.forward(xin, hidden)
        if random.random() < epsilon:
            return random.randint(0,1), output
        else:
            return output[0].argmax().item(), output
    
    def init_hidden_state(self, batch_size=None, training=None):
        assert training is not None, "training step parameter should be determined"
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_size])
        else:
            return torch.zeros([1, 1, self.hidden_size])
        
    def checkpoint_weights(self):
        self.saved_weights = pickle.loads(pickle.dumps(self.state_dict()))
        return self.saved_weights
    
    def save_weights_to_path(self, path, weights=None):
        torch.save(self.state_dict() if weights is None else weights, path)

    def load_weights_from_path(self, path):
        self.load_state_dict(torch.load(path))
