import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, seed, arch_params):
        """ arch_parameters is a dictionary like:
        {'state_and_action_sizes' : (num1, num2), 'Linear_1' : layer_size_1,..,'Linear_n' : layer_size_n}
        """
        super(QNetwork, self).__init__()
        self.seed_as_int = seed
        torch.manual_seed(seed)
        self.arch_params = arch_params

        prev_layer_size = arch_params['state_and_action_sizes'][0]
        keys = list(arch_params.keys())
        list_of_layers = []
        for i in range(len(self.arch_params)):
            key = keys[i]
            if key == 'state_and_action_sizes':  # the first element of arch_params is (state_size, action_size) tuple
                continue
            layer_type = key.split('_')[0]
            if layer_type == 'Linear':
                layer_size = arch_params[key]
                list_of_layers.append(nn.Linear(prev_layer_size, layer_size))
                prev_layer_size = layer_size
            elif layer_type == 'ReLU':
                list_of_layers.append(nn.ReLU())
            else:
                raise AttributeError("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
        list_of_layers.append(nn.Linear(arch_params[keys[-3]], 1))  # add layer to calculate value function

        self.layers = nn.ModuleList(list_of_layers)

    def forward(self, state):  # get action values
        """Build a network that maps state -> action values."""
        x = state
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if i == (len(self.layers) - 3):
                y = self.layers[-1](x)

        return y+(x-x.mean()) # y - value function

    def save(self, save_to):
        file = {'arch_params': self.arch_params,
                'state_dict': self.state_dict()}
        torch.save(file, save_to)

    def load(self, load_from):
        checkpoint = torch.load(load_from)
        self.__init__(self.seed_as_int, checkpoint['arch_params'])
        self.load_state_dict(checkpoint['state_dict'])
        return self
