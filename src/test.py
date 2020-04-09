import model.rbc_model
import micromodels.net_0.model
import torch 
import numpy


class Loss:

    def init(self, mesh):
        self.initial_length  = mesh.length().detach()
        self.initial_surface = mesh.surface().detach()
        self.initial_volume  = mesh.volume().detach()

    def forward(self, mesh):
        volume      = mesh.volume()
        length      = mesh.length()

        '''
        - minimal volume
        - length conservation law
        '''
        loss = 0.1*volume + ((self.initial_length - length)**2.0) 

        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rbc = model.rbc_model.RbcModel("objs/sphere_86.obj", micromodels.net_0.model, Loss, device)

optimizer  = torch.optim.Adam(rbc.triangle_micromodel.parameters(), lr= 0.01)  

loss_best = 100.0
steps     = 1000
for step in range(steps):
    #initial state
    
    angle = list(numpy.random.rand(3)*2*3.141592654)
    angle[2] = 0

    position = 0.1*(numpy.random.rand(3) - 0.5)*2.0
    rbc.init(initial_position= position, initial_angle = angle)

    #perform some simulation steps
    for i in range(128):
        rbc.forward(dt = 0.01)

    #compute loss
    loss = rbc.get_loss()

    loss.backward()
    optimizer.step()

    print("step : ", step, ", loss = ", loss.detach().to("cpu").numpy())

    if loss < loss_best:
        loss_best = loss
        rbc.triangle_micromodel.save("micromodels/net_0/")
        print("saving new best model\n")

'''
rbc.triangle_micromodel.load("micromodels/net_0/")


rbc.init()
for i in range(128):
    rbc.forward(dt = 0.01)

loss = rbc.get_loss()
rbc.mesh_model.plot()
'''