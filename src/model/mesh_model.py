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

    def init(self, initial_position = [0.0, 0.0, 0.0], initial_velocity = [0.0, 0.0, 0.0], initial_angle = [0.0, 0.0, 0.0]):
        self.points_count     = len(self.model.points)
        self.triangles_count  = len(self.model.polygons)

        self.initial_position = torch.zeros((self.points_count, 3)).to(self.device) 
        self.initial_velocity = torch.zeros((self.points_count, 3)).to(self.device) 
        self.initial_force    = torch.zeros((self.points_count, 3)).to(self.device) 
       
        for i in range(self.points_count):
            self.initial_position[i] = torch.from_numpy( self._rotate(self.model.points[i], initial_angle) + initial_position)

        for i in range(self.points_count):
            self.initial_velocity[i] = torch.from_numpy(numpy.array( self._rotate(initial_velocity, initial_angle) ))


        self.position = self.initial_position.clone().to(self.device) 
        self.velocity = self.initial_velocity.clone().to(self.device) 
        self.force    = self.initial_force.clone().to(self.device) 


        '''
        state tensor shape :
        0 - total triangles count
        1 - points per triangle; (3)
        2 - state of point : position, velocity, force; (3)
        3 - point elements : x, y, z; (3)
        '''

        self.state_tensor = self.create_state()

    def create_state(self):
        state_tensor = torch.zeros((self.triangles_count, 3, 3, 3), requires_grad=True).to(self.device)

        center = self.center()

        #this is bottle neck - need to refactoring for faster run
        for j in range(self.triangles_count):
            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]

            
            state_tensor[j][0][0] = self.position[p0_idx] - center
            state_tensor[j][0][1] = self.velocity[p0_idx]
            state_tensor[j][0][2] = self.force[p0_idx]

            state_tensor[j][1][0] = self.position[p1_idx] - center
            state_tensor[j][1][1] = self.velocity[p1_idx]
            state_tensor[j][1][2] = self.force[p1_idx]

            state_tensor[j][2][0] = self.position[p2_idx] - center
            state_tensor[j][2][1] = self.velocity[p2_idx]
            state_tensor[j][2][2] = self.force[p2_idx]
        
        return state_tensor.detach()


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
                
    def plot(self, file_name = None):
        fig = pyplot.figure()
        ax = Axes3D(fig)

        position = self.position.detach().to("cpu").numpy()
        position = numpy.transpose(position)

        ax.scatter(position[0], position[1], position[2])

            
        for j in range(self.triangles_count):

            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]

            x = [ position[0][p0_idx], position[0][p1_idx], position[0][p2_idx], position[0][p0_idx] ]
            y = [ position[1][p0_idx], position[1][p1_idx], position[1][p2_idx], position[1][p0_idx] ]
            z = [ position[2][p0_idx], position[2][p1_idx], position[2][p2_idx], position[2][p0_idx] ]
        
            ax.plot3D(x, y, z, c = "black") 
        

        ax.set_xlim([-1.0,1.0])
        ax.set_ylim([-1.0,1.0])
        ax.set_zlim([-1.0,1.0])
        
        if file_name == None:
            pyplot.show()
        else:
            pyplot.savefig(file_name)
            pyplot.close()

    def center(self):
        result = torch.mean(self.position, dim = 0)
        return result

    def volume(self):
        center = self.center()
        volume = torch.norm(self.position - center, dim = 1)**3
        return volume.mean()

    def surface(self):
        s = torch.zeros(self.triangles_count).to(self.device)

        for j in range(self.triangles_count):
            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
 
            a = self.position[p0_idx]
            b = self.position[p1_idx]

            s[j] = 0.5*torch.norm(torch.cross(a, b))

        return s.mean()

    def length(self):
        result = torch.zeros(self.triangles_count).to(self.device)

        for j in range(self.triangles_count):

            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]
             
            a = self.position[p0_idx]
            b = self.position[p1_idx]
            c = self.position[p2_idx]

            ab = (a - b).clone()
            bc = (b - c).clone()
            ca = (c - a).clone()
           
            result[j] = (torch.norm(ab) + torch.norm(bc) + torch.norm(ca))/3.0
        return result.mean()






    def _group_forces(self, force):
        result = torch.rand((self.points_count, 3)).to(self.device)

        '''
        TODO -> this is BRUTAL bottleneck
        '''
        for j in range(self.triangles_count):
            p0_idx = self.model.polygons[j][0]
            p1_idx = self.model.polygons[j][1]
            p2_idx = self.model.polygons[j][2]

            result[p0_idx] = result[p0_idx] + force[j]
            result[p1_idx] = result[p1_idx] + force[j]
            result[p2_idx] = result[p2_idx] + force[j]
        

        return result


    def _rotate(self, point, angle):
        yaw     = angle[0]
        pitch   = angle[1]
        roll    = angle[2]

        Rz = numpy.zeros((3, 3))
        Rz[0][0] = numpy.cos(yaw)
        Rz[0][1] = -numpy.sin(yaw)
        Rz[1][0] = numpy.sin(yaw)
        Rz[1][1] = numpy.cos(yaw)
        Rz[2][2] = 1.0

        Ry = numpy.zeros((3, 3))
        Ry[0][0] = numpy.cos(pitch)
        Ry[0][2] = numpy.sin(pitch)
        Ry[1][1] = 1.0
        Ry[2][0] = -numpy.sin(pitch)
        Ry[2][2] = numpy.cos(pitch)
       
        Rx = numpy.zeros((3, 3))
        Rx[0][0] = 1.0
        Rx[1][1] = numpy.cos(roll)
        Rx[1][2] = -numpy.sin(roll)
        Rx[2][1] = numpy.sin(roll)
        Rx[2][2] = -numpy.cos(roll)

        R = numpy.matmul(numpy.matmul(Rz, Ry), Rx)
        result = numpy.dot(numpy.asarray(point), R)        
        
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