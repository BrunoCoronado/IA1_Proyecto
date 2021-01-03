from FileManagement import File
from Logistic_Regression.Data import Data
from Logistic_Regression.Model import Model
from Logistic_Regression import Plotter
import numpy as np

train_set_usac, test_set_usac = File.obtenerImagenes('USAC')
train_set_landivar, test_set_landivar = File.obtenerImagenes('Landivar')
train_set_mariano, test_set_mariano = File.obtenerImagenes('Mariano')
train_set_marroquin, test_set_marroquin = File.obtenerImagenes('Marroquin')

model1 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.001, lam=150, it=600)
model2 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.002, lam=200, it=800)
model3 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.0015, lam=200, it=1000)
model4 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.0025, lam=175, it=655)
model5 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.003, lam=210, it=145)
entrenamiento1, validacion1 = model1.entrenar()
entrenamiento2, validacion2 = model2.entrenar()
entrenamiento3, validacion3 = model3.entrenar()
entrenamiento4, validacion4 = model4.entrenar()
entrenamiento5, validacion5 = model5.entrenar()
Plotter.guardarModelo([model1, model2, model3, model4, model5], 'USAC.png')

model6 = Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.0014, lam=100, it=650)
model7 = Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.001444, lam=200, it=700)
model8 = Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.00225, lam=300, it=800)
model9 = Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.001115, lam=400, it=825)
model10 = Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.000003, lam=500, it=700)
entrenamiento6, validacion7 = model6.entrenar()
entrenamiento7, validacion7 = model7.entrenar()
entrenamiento8, validacion8 = model8.entrenar()
entrenamiento9, validacion9 = model9.entrenar()
entrenamiento10, validacion10 = model10.entrenar()
Plotter.guardarModelo([model6, model7, model8, model9, model10], 'Landivar.png')

model11 = Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.00001, lam=150, it=600)
model12 = Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.0025, lam=250, it=700)
model13 = Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.0010, lam=350, it=600)
model14 = Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.002555, lam=375, it=700)
model15 = Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.001, lam=400, it=600)
entrenamiento11, validacion11 = model11.entrenar()
entrenamiento12, validacion12 = model12.entrenar()
entrenamiento13, validacion13 = model13.entrenar()
entrenamiento14, validacion14 = model14.entrenar()
entrenamiento15, validacion15 = model15.entrenar()
Plotter.guardarModelo([model11, model12, model13, model14, model15], 'Mariano.png')

model16 = Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.003, lam=150, it=650)
model17 = Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.0001, lam=200, it=700)
model18 = Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.0018, lam=200, it=800)
model19 = Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.001, lam=175, it=825)
model20 = Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.00014, lam=210, it=700)
entrenamiento16, validacion16 = model16.entrenar()
entrenamiento17, validacion17 = model17.entrenar()
entrenamiento18, validacion18 = model18.entrenar()
entrenamiento19, validacion19 = model19.entrenar()
entrenamiento20, validacion20 = model20.entrenar()
Plotter.guardarModelo([model16, model17, model18, model19, model20], 'Marroquin.png')

