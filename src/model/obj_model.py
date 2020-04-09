import numpy

class ObjModel:
    def __init__(self, file_name):

        self.points = []
        self.polygons = []

        fp = open(file_name, "r")

        line = fp.readline()
        while line:
            if line.find("v ") != -1:
                self._parse_vertice(line)
            if line.find("f ") != -1:
                self._parse_polygons(line)

            line = fp.readline()
        fp.close()

        self.points = numpy.asarray(self.points, dtype=float)

    def _parse_vertice(self, line):
        splitted = line.split()
        x = float(splitted[1])
        y = float(splitted[2])
        z = float(splitted[3])

        self.points.append([x, y, z])

    def _parse_polygons(self, line):
        splitted = line.split()

        points = []

        for i in range(len(splitted) - 1):
            p = int(splitted[i+1].split("/")[0]) - 1
            points.append(p)
        
        self.polygons.append(points)


if __name__ == "__main__":
    obj_model = ObjModel("sphere_86.obj")
