import random
import pickle
from numpy.random import default_rng
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class DRQN(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 recurrent_cell='GRU'):
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.recurrent_cell = recurrent_cell.lower()
        if self.recurrent_cell == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        elif self.recurrent_cell == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        else:
            raise Exception("Invalid recurrent_cell type. Supported options: ['rnn', 'gru']")
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.initial_weights = self.checkpoint_weights()
        self.rng = default_rng()

    def forward(self, xin, hidden):
        z, _ = self.rnn(xin, hidden)
        return self.output(z), z
    
    def reset_rng(self, seed):
        self.rng = default_rng(seed)

    def sample_action(self, xin, hidden, epsilon=None, tau=None):
        output = self.forward(xin, hidden)
        if epsilon is not None: # epsilon-greedy policy
            if self.rng.random() < epsilon: # choose random action
                return int(self.rng.random() < 0.5), output
            else: # choose best action
                return output[0].argmax().item(), output
        elif tau is not None: # softmax policy
            logits = nn.functional.softmax(output[0]/tau, dim=-1)
            return Categorical(logits=logits).sample().item(), output
    
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


    def initialize(self, gain=1):
        """
        https://github.com/rodrigorivera/mds20_replearning/blob/0426340725fd55a616b0d40356ddcebe06ed0f24/skip_thought_vectors/encoder.py
        https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/2
        https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5
        https://pytorch.org/docs/stable/nn.init.html
        """
        assert self.recurrent_cell.lower() in ['gru', 'rnn']
        for weight_ih, weight_hh, bias_ih, bias_hh in self.rnn.all_weights:
            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)
            for i in range(0, weight_hh.size(0), self.hidden_size):
                nn.init.orthogonal_(weight_hh.data[i:(i+self.hidden_size)], gain=gain) # orthogonal
                nonlinearity = 'tanh' if ((self.recurrent_cell.lower() == 'rnn') or (i == 2)) else 'sigmoid'
                nn.init.xavier_uniform_(weight_ih.data[i:(i+self.hidden_size)], gain=nn.init.calculate_gain(nonlinearity)) # glorot_uniform

    def reset(self, gain=None):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
                if gain != None and layer == self.rnn:
                    self.initialize(gain=gain)
                else:
                    layer.reset_parameters()
        self.initial_weights = self.checkpoint_weights()
