from flask import request, jsonify
from flask_api import FlaskAPI
from flask_cors import CORS, cross_origin
from FileManagement import File
from Logistic_Regression.Data import Data
from Logistic_Regression.Model import Model
from Logistic_Regression import Plotter
import numpy as np
from datetime import datetime
import base64
from PIL import Image

app = FlaskAPI(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


modelo_usac = None
modelo_landivar = None
modelo_mariano = None
modelo_marroquin = None

@app.route("/analizar", methods=['POST'])
@cross_origin()
def generar():
    """
    Analizar imagen que viene en base64
    """
    
    if modelo_usac == None:
        inicializarModelos()

    # global criterio
    # global seleccion
    # global data

    with open("./tmp.jpg", "wb") as fh:
        fh.write(base64.b64decode(request.json['imagen']))
        fh.close()

    image = Image.open('./tmp.jpg')
    data = np.append(np.asarray(image).reshape(-1), [1]) / 255
    resultados = []
    resultados.append(modelo_usac.predecir(data))
    resultados.append(modelo_landivar.predecir(data))
    resultados.append(modelo_mariano.predecir(data))
    resultados.append(modelo_marroquin.predecir(data))
    print(resultados)

    return jsonify(
        resultados = resultados,
        mensaje = 'imagen Analizada!',
        status = 200
    )

def verModelos():
    print('USAC -> Entrenamiento: ' +  '%.2f' % modelo_usac.train_accuracy + ' Validacion: ' + '%.2f' % modelo_usac.test_accuracy)
    print('Landivar -> Entrenamiento: ' +  '%.2f' % modelo_landivar.train_accuracy + ' Validacion: ' + '%.2f' % modelo_landivar.test_accuracy)
    print('Mariano -> Entrenamiento: ' +  '%.2f' % modelo_mariano.train_accuracy + ' Validacion: ' + '%.2f' % modelo_mariano.test_accuracy)
    print('Marroquin -> Entrenamiento: ' +  '%.2f' % modelo_marroquin.train_accuracy + ' Validacion: ' + '%.2f' % modelo_marroquin.test_accuracy)

def guardarBitacora(modelos, universidad):
    f = open("bitacora.txt", "a+")
    now = datetime.now()
    f.write('--------------------------------------\n')
    f.write(now.strftime("%d/%m/%Y %H:%M:%S") + '\n')
    index = 0
    while index < 5:
        f.write( universidad + ' - MODELO ' + str(index + 1) + '\n')
        f.write(' Entrenamiento: ' +  '%.2f' % modelos[index].train_accuracy + '%\n')
        f.write(' Validacion: ' + '%.2f' % modelos[index].test_accuracy + '%\n')
        index += 1

    f.write('--------------------------------------\n')
    f.close()
    return

def inicializarModelos():
    global modelo_usac
    global modelo_landivar
    global modelo_mariano
    global modelo_marroquin

    train_set_usac, test_set_usac = File.obtenerImagenes('USAC')
    train_set_landivar, test_set_landivar = File.obtenerImagenes('Landivar')
    train_set_mariano, test_set_mariano = File.obtenerImagenes('Mariano')
    train_set_marroquin, test_set_marroquin = File.obtenerImagenes('Marroquin')

    modelos_usac = []
    modelos_landivar = []
    modelos_mariano = []
    modelos_marroquin = []

    modelos_usac.append(Model(train_set_usac, test_set_usac, reg=False, alpha=0.001, lam=150, it=600))
    modelos_usac.append(Model(train_set_usac, test_set_usac, reg=False, alpha=0.002, lam=200, it=800))
    modelos_usac.append(Model(train_set_usac, test_set_usac, reg=False, alpha=0.000015, lam=200, it=1000))
    modelos_usac.append(Model(train_set_usac, test_set_usac, reg=False, alpha=0.0025, lam=175, it=655))
    modelos_usac.append(Model(train_set_usac, test_set_usac, reg=False, alpha=0.003, lam=210, it=145))
    for m in modelos_usac:
        m.entrenar()
    Plotter.guardarModelo(modelos_usac, 'USAC.png')
    guardarBitacora(modelos_usac, 'USAC')
    modelo_usac = sorted(modelos_usac, key=lambda item: item.test_accuracy, reverse=True)[0]

    modelos_landivar.append(Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.0014, lam=100, it=650))
    modelos_landivar.append(Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.001444, lam=200, it=700))
    modelos_landivar.append(Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.000015, lam=100, it=900))
    modelos_landivar.append(Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.0015, lam=400, it=825))
    modelos_landivar.append(Model(train_set_landivar, test_set_landivar, reg=False, alpha=0.0000033, lam=500, it=700))
    for m in modelos_landivar:
        m.entrenar()
    Plotter.guardarModelo(modelos_landivar, 'Landivar.png')
    guardarBitacora(modelos_landivar, 'Landivar')
    modelo_landivar = sorted(modelos_landivar, key=lambda item: item.test_accuracy, reverse=True)[0]

    modelos_mariano.append(Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.00001, lam=150, it=1200))
    modelos_mariano.append(Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.0025, lam=250, it=700))
    modelos_mariano.append(Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.0010, lam=350, it=900))
    modelos_mariano.append(Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.002555, lam=375, it=700))
    modelos_mariano.append(Model(train_set_mariano, test_set_mariano, reg=False, alpha=0.001, lam=100, it=600))
    for m in modelos_mariano:
        m.entrenar()
    Plotter.guardarModelo(modelos_mariano, 'Mariano.png')
    guardarBitacora(modelos_mariano, 'Mariano')
    modelo_mariano = sorted(modelos_mariano, key=lambda item: item.test_accuracy, reverse=True)[0]

    modelos_marroquin.append(Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.003, lam=150, it=650))
    modelos_marroquin.append(Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.0001, lam=200, it=700))
    modelos_marroquin.append(Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.0018, lam=200, it=800))
    modelos_marroquin.append(Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.001, lam=175, it=825))
    modelos_marroquin.append(Model(train_set_marroquin, test_set_marroquin, reg=False, alpha=0.00014, lam=210, it=700))
    for m in modelos_marroquin:
        m.entrenar()
    Plotter.guardarModelo(modelos_marroquin, 'Marroquin.png')    
    guardarBitacora(modelos_marroquin, 'Marroquin')
    modelo_marroquin = sorted(modelos_marroquin, key=lambda item: item.test_accuracy, reverse=True)[0]

    verModelos()


if __name__ == "__main__":
    app.run(debug=True)
