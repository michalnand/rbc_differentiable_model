from model.obj_model import ObjModel
import torch
import numpy

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class MeshModel:
    def __init__(self, file_name, device = "cpu"):
        self.model = ObjModel(file_name)
        self.device = device

        self.init()

    def init(self, max_time_steps = 10, initial_position = [0.0, 0.0, 0.0], initial_velocity = [0.0, 0.0, 0.0]):
        self.points_count     = len(self.model.points)
        self.triangles_count  = len(self.model.polygons)

        self.initial_position = torch.zeros((self.points_count, 3))
        self.initial_velocity = torch.zeros((self.points_count, 3))
        self.initial_force    = torch.zeros((self.points_count, 3))

       
        for i in range(self.points_count):
            self.initial_position[i] = torch.from_numpy(self.model.points[i] + initial_position)

        for i in range(self.points_count):
            self.initial_velocity[i] = torch.from_numpy(numpy.array(initial_velocity))

       
        self.initial_position.to(self.device) 
        self.initial_velocity.to(self.device)
        self.initial_force.to(self.device)

        self.position = self.initial_position.clone()
        self.velocity = self.initial_velocity.clone()
        self.force    = self.initial_force.clone()

        '''
        tensor shape :
        0 - total triangles count
        1 - points per triangle; (3)
        2 - state of point : position, velocity, force; (3)
        3 - point elements : x, y, z; (3)
        '''
        self.state_tensor = torch.zeros((self.triangles_count, 3, 3, 3)).to(self.device)
        self.create_state_tensor()

    def create_state_tensor(self):
        #this is bottle neck - need to refactoring for faster run
        for j in range(self.triangles_count):
            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]
            
            self.state_tensor[j][0][0] = self.position[p0_idx]
            self.state_tensor[j][0][1] = self.velocity[p0_idx]
            self.state_tensor[j][0][2] = self.force[p0_idx]

            self.state_tensor[j][1][0] = self.position[p1_idx]
            self.state_tensor[j][1][1] = self.velocity[p1_idx]
            self.state_tensor[j][1][2] = self.force[p1_idx]

            self.state_tensor[j][2][0] = self.position[p2_idx]
            self.state_tensor[j][2][1] = self.velocity[p2_idx]
            self.state_tensor[j][2][2] = self.force[p2_idx]

        return self.state_tensor

    def update_state(self, force, dt = 0.01, position_clip = 10.0, velocity_clip = 10.0, force_clip = 10.0):
        force_ = self._group_forces(force)
        self.force = torch.clamp(force_, -force_clip, force_clip)
        
        ''' 
        euler method for dif equation solving
        TODO : Runge–Kutta, for faster convergence
        '''
        self.velocity = torch.clamp(self.velocity + self.force*dt, -velocity_clip, velocity_clip)
        self.position = torch.clamp(self.position + self.velocity*dt, -position_clip, position_clip)

        return self.position, self.velocity, self.force
                
    def plot(self):
        fig = pyplot.figure()
        ax = Axes3D(fig)

        position = self.position.detach().to("cpu").numpy()
        position = numpy.transpose(position)
        ax.scatter(position[0], position[1], position[2])
        
        pyplot.show()

    def center(self):
        result = torch.mean(self.position, dim = 0)
        return result

    def volume(self):
        center = self.center()
        dif    = self.position - center

        length = torch.norm(dif, dim = 1)

        v = (length**3).mean()
        
        return v

    def surface(self):
        s = torch.zeros(self.triangles_count).to(self.device)
        return s.mean()

    def length(self):
        l = torch.zeros(self.triangles_count).to(self.device)

        '''
        for j in range(self.points_count):
            a = self.position[j][0][0]
            b = self.position[j][1][0]
            c = self.position[j][2][0]

            tmp = torch.norm(a - b)
            tmp+= torch.norm(b - c)
            tmp+= torch.norm(c - a)

            l[j] = tmp/3.0
        '''

        return l.mean()






    def _group_forces(self, force):
        result = torch.rand((self.points_count, 3)).to(self.device)

        '''
        TODO -> this is BRUTAL bottleneck
        '''
        for j in range(self.triangles_count):
            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]

            result[p0_idx]+= force[j]
            result[p1_idx]+= force[j]
            result[p2_idx]+= force[j]
        

        return result


if __name__ == "__main__":
    mesh_model = MeshModel("sphere_86.obj")
    
    steps = 16
    for i in range(steps):
        force = torch.rand((mesh_model.triangles_count, 3))
        mesh_model.update_state(force)

    mesh_model.plot()


    print("volume = ", mesh_model.volume())
    print("length = ", mesh_model.length())