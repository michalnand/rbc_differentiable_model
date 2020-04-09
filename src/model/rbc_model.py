import model.mesh_model

import torch

class RbcModel:
    def __init__(self, mesh_file_name, TriangleMicromodel, Loss, device = "cpu"):
        self.mesh_model         = model.mesh_model.MeshModel(mesh_file_name, device)

        input_shape     = self.mesh_model.state_tensor.shape[1:]
        outputs_count   = 3
        self.triangle_micromodel    = TriangleMicromodel.Create(input_shape, outputs_count, device)
        self.loss                   = Loss()

        self.init()

    def init(self, initial_position = [0.0, 0.0, 0.0], initial_velocity = [0.0, 0.0, 0.0], initial_angle = [0.0, 0.0, 0.0]):
        self.mesh_model.init(initial_position, initial_velocity, initial_angle)
        self.loss.init(self.mesh_model)

    def forward(self, dt = 0.01):
        
        '''
        create state tensor, for each point to point interraction
        '''
        state = self.mesh_model.state_tensor

        '''
        apply model step, for each point
        '''
        d_state = self.triangle_micromodel.forward(state.detach())

        '''
        solve mesh diferential equations
        '''
        self.mesh_model.update_state(d_state, dt)

        '''
        compute loss
        '''
        loss = self.loss.forward(self.mesh_model)

        return loss

    def get_loss(self):
        return self.loss.forward(self.mesh_model)


    def forward_steps(self, steps, dt = 0.01):
        loss = 0.0
        for n in range(steps):
            loss+= self.forward(dt)

        return loss


