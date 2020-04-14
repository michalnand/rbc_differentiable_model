import model.rbc_model
import micromodels.net_0.model
import torch 
import numpy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rbc = model.rbc_model.RbcModel("objs/sphere_86.obj", micromodels.net_0.model, Loss, device)

rbc.triangle_micromodel.load("micromodels/net_0/")
rbc.triangle_micromodel.eval()

rbc.init()
for i in range(256):
    rbc.forward(dt = 0.01)

    rbc.mesh_model.plot("images/step_" + str(i) + ".png")

rbc.mesh_model.plot()
