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
        length      = mesh.length()
        surface     = mesh.surface()
        volume      = mesh.volume()

        loss_surface  = -surface
        loss_volume   = (self.initial_volume*0.25 - volume)**2.0

        loss =  loss_volume + loss_surface
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rbc = model.rbc_model.RbcModel("objs/sphere_86.obj", micromodels.net_0.model, Loss, device)

optimizer  = torch.optim.Adam(rbc.triangle_micromodel.parameters(), lr= 0.01)  

loss_best   = 100.0
loss_smooth = 100.0

steps     = 1000
for step in range(steps):
    #initial state
    rbc.init(position_noise_level = 0.1, velocity_noise_level = 0.01)

    #perform some simulation steps
    for i in range(128):
        rbc.forward(dt = 0.01)

    #compute loss
    loss = rbc.get_loss()

    loss.backward()
    optimizer.step()

    loss_np = loss.detach().to("cpu").numpy()
    print("step : ", step, ", loss = ", loss_np)

    loss_smooth = 0.9*loss_smooth + 0.1*loss_np
    if loss_smooth < loss_best:
        loss_best = loss_smooth
        rbc.triangle_micromodel.save("micromodels/net_0/")
        print("saving new best model\n")
