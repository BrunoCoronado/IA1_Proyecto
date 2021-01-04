class Nodo:
    def __init__(self, solucion = [], fitness = 0, nn = None):
        self.solucion = solucion
        self.fitness = fitness
        self.nn = nn