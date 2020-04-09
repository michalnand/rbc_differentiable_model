import numpy
import matplotlib.pyplot as plt
import torch

class Wave:
    def __init__(self, noise = 0.3, length = 256, positive = True):

        self.result = numpy.zeros(length)

        pi = 3.141592654

        center      = 0.5 + 0.3*(numpy.random.random()*2.0 - 1.0)
        shape       = 5.0 + 10.0*numpy.random.random()

        for i in range(length):
                
            x       = i/length

            if positive:
                y_ = numpy.sin(3.0*(x + center)*2.0*pi)*numpy.exp(-shape*(x - center)**2)
            else:
                y_ = 0.0
            y  = (1.0 - noise)*y_ + noise*numpy.random.randn()
        
            self.result[i] = y
        
      



class Create:
    def __init__(self, training_count = 1000, testing_count = 1000):
 
        self.channels = 1
        self.width   = 256

        self.input_shape   = (1, self.width)
        self.classes_count = 2

        self.training_x, self.training_y = self.generate(training_count)
        self.testing_x, self.testing_y   = self.generate(training_count)

        

    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y)

    def generate(self, count):
        x_input     = numpy.zeros((count, self.width))
        y_target    = numpy.zeros((count, 2))

        for n in range(count):  
            if n%2 == 0:
                y_target[n][0] = 1.0

                w = Wave(0.05, self.width, False)
            else:
                y_target[n][1] = 1.0

                w = Wave(0.05, self.width, True)

            x_input[n] = w.result


        return x_input, y_target


    def _get_batch(self, x, y, batch_size = 32):
        result_x = torch.zeros((batch_size, 1, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size):
            idx = numpy.random.randint(len(x))

            result_x[i][0] = torch.from_numpy(x[idx])
            result_y[i] = torch.from_numpy(y[idx])

        return result_x, result_y

if __name__ == "__main__":
    dataset = Create(100, 100)

    x, y = dataset.get_training_batch()
    print(x.shape, y.shape)