import torch
import torch.nn as nn
import numpy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Create(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs_count = numpy.prod(numpy.array(input_shape))  

        self.layers = [ 
                        Flatten(),
                        nn.Linear(inputs_count, 32),
                        nn.ReLU(),
                        nn.Linear(32, outputs_count)
                    ]

        '''
        self.layers = [ 
                        Flatten(),
                        nn.Linear(inputs_count, outputs_count)
                    ]
        '''

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("\n\n")
        print("network model = ")
        print(self.model)

    def forward(self, state):
        return self.model.forward(state)

  
    def save(self, path):
        name = path + "trained/model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name, map_location = self.device))
        self.model.eval() 