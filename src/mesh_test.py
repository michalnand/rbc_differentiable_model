import numpy
import torch 

import model.mesh_model



device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mesh_model  = model.mesh_model.MeshModel("objs/sphere_86.obj", device)

#random points seeding, gaussian_noise, mean = 0.0, var = 1.0, aplitude = noise
mesh_model.init(position_noise_level = 0.05, velocity_noise_level = 0.01)

state = mesh_model.create_state(relative_state=True)

print("center position = ", mesh_model.center_position)
print("center velocity = ", mesh_model.center_velocity)
print("center force = ", mesh_model.center_force)
print("\n\n\n")

print("length       = ", mesh_model.length())
print("surface      = ", mesh_model.surface())
print("volume       = ", mesh_model.volume())
print("curvature    = ", mesh_model.curvature())



mesh_model.plot()