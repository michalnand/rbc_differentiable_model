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

        loss_length   = (self.initial_length - length)**2.0
        loss_surface  = -surface
        loss_volume   = volume 

        loss =  loss_volume + loss_surface + loss_length

        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rbc = model.rbc_model.RbcModel("objs/sphere_86.obj", micromodels.net_0.model, Loss, device)

rbc.triangle_micromodel.load("micromodels/net_0/")
rbc.triangle_micromodel.eval()

rbc.init()
for i in range(128):
    rbc.forward(dt = 0.01)

    rbc.mesh_model.plot("images/step_" + str(i) + ".png")

    print("step = ", i)

rbc.mesh_model.plot()
