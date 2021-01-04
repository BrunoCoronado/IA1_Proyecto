from Util.ReadFile import cargarDatos
from Util import Plotter
from Util.Nodo import Nodo
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
import numpy as np

    #alpha,    lambda, m_it, kp
hiperparametros = [
    [0.1,      0.001,  1000, 1    ],
    [0.001,    0.7,    1250, 0.5  ],
    [0.0001,   0.8,    2000, 0.67 ],
    [0.0015,   0.005,  500,  0.9  ],
    [0.002,    0.015,  400,  0.1  ],
    [0.02,     0.152,  100,  0.55 ],
    [0.000001, 1,      2500, 0.03 ],
    [0.005,    0.0335, 1500, 0.98 ],
    [0.03,     1.12,   800,  0.4  ],
    [0.0358,   0,      650,  0.564],
]

train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = cargarDatos()

train_set = Data(train_set_x, train_set_y)
val_set = Data(val_set_x, val_set_y)
test_set = Data(test_set_x, test_set_y)

capas = [train_set.n, 10, 5, 3, 1]

nn = NN_Model(train_set, capas, alpha=0.0358, lambd=0.152, iterations=1500, keep_prob=1)
# nn = NN_Model(train_set, capas, alpha=0.001, lambd=0.152, iterations=1500, keep_prob=1)
nn.training(False)
Plotter.show_Model([nn])

print('Entrenamiento Modelo 1')
nn.predict(train_set, True)
print('Validacion Modelo 1')
nn.predict(val_set, True)
print('Pruebas Modelo 1')
nn.predict(test_set, True)
