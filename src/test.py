import model.rbc_model
import micromodels.net_0.model
import torch

#torch.autograd.set_detect_anomaly(True)

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
        loss = volume + ((self.initial_length - length)**2.0) 

        return loss



rbc = model.rbc_model.RbcModel("objs/sphere_86.obj", micromodels.net_0.model, Loss)

optimizer  = torch.optim.Adam(rbc.triangle_micromodel.parameters(), lr= 0.1, weight_decay=10**-5)  

rbc.init()
loss = 0
for i in range(1):
    loss+= rbc.forward(dt = 0.01)

loss.backward()
optimizer.step()

print("loss = ", loss.detach().to("cpu").numpy())

'''
rbc.init()
for i in range(100):
    rbc.forward(dt = 0.01)

rbc.mesh_model.plot()
'''