from flask import request, jsonify
from flask_api import FlaskAPI
from flask_cors import CORS, cross_origin
from Util.ReadFile import cargarDatos
from Util import Plotter
from Util.Nodo import Nodo
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
import numpy as np
import numpy as np

app = FlaskAPI(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

modelo_nn = None

hiperparametros = [
    [0.1,  0.001,  1000,  1    ],
    [0.2,  0.3,    1250,  0.95 ],
    [0.3,  0.5,    2000,  0.90 ],
    [0.5,  0.005,  1100,  0.85 ],
    [0.03, 0.015,  1050,  0.83 ],
    [0.25, 0.152,  2250,  0.70 ],
    [0.14, 0.001,  2500,  0.75 ],
    [0.28, 0.0335, 1800,  0.99 ],
    [0.44, 0.12,   1300,  0.93 ],
    [0.04, 0,      3000,  0.88 ],
]

#train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = cargarDatos()
train_set_x, train_set_y, val_set_x, val_set_y = cargarDatos()

train_set = Data(train_set_x, train_set_y)
val_set = Data(val_set_x, val_set_y)
#test_set = Data(test_set_x, test_set_y)

capas = [train_set.n, 15, 10, 7, 1]

@app.route("/consultar", methods=['POST'])
@cross_origin()
def consultar():
    """
    Predecir con la red neuronal
    """
    
    if modelo_nn == None:
        inicializarModelo()

    # global criterio
    # global seleccion
    # global data

    # with open("./tmp.jpg", "wb") as fh:
    #     fh.write(base64.b64decode(request.json['imagen']))
    #     fh.close()

    # image = Image.open('./tmp.jpg')
    # data = np.append(np.asarray(image).reshape(-1), [1]) / 255
    # resultados = []
    # resultados.append(modelo_usac.predecir(data))
    # resultados.append(modelo_landivar.predecir(data))
    # resultados.append(modelo_mariano.predecir(data))
    # resultados.append(modelo_marroquin.predecir(data))
    # print(resultados)

    return jsonify(
        # resultados = resultados,
        mensaje = 'imagen Analizada!',
        status = 200
    )

@app.route("/catalogos", methods=['GET'])
@cross_origin()
def catalogos():
    """
    Obtiene todos los catalogos
    """

    # global criterio
    # global seleccion
    # global data

    # with open("./tmp.jpg", "wb") as fh:
    #     fh.write(base64.b64decode(request.json['imagen']))
    #     fh.close()

    # image = Image.open('./tmp.jpg')
    # data = np.append(np.asarray(image).reshape(-1), [1]) / 255
    # resultados = []
    # resultados.append(modelo_usac.predecir(data))
    # resultados.append(modelo_landivar.predecir(data))
    # resultados.append(modelo_mariano.predecir(data))
    # resultados.append(modelo_marroquin.predecir(data))
    # print(resultados)

    return jsonify(
        # resultados = resultados,
        mensaje = 'imagen Analizada!',
        status = 200
    )

@app.route("/hiperparametros", methods=['GET'])
@cross_origin()
def hiperparam():
    """
    Obtiene los hiperparametros
    """

    # global criterio
    # global seleccion
    # global data

    # with open("./tmp.jpg", "wb") as fh:
    #     fh.write(base64.b64decode(request.json['imagen']))
    #     fh.close()

    # image = Image.open('./tmp.jpg')
    # data = np.append(np.asarray(image).reshape(-1), [1]) / 255
    # resultados = []
    # resultados.append(modelo_usac.predecir(data))
    # resultados.append(modelo_landivar.predecir(data))
    # resultados.append(modelo_mariano.predecir(data))
    # resultados.append(modelo_marroquin.predecir(data))
    # print(resultados)

    return jsonify(
        # resultados = resultados,
        mensaje = 'imagen Analizada!',
        status = 200
    )

def inicializarModelo():
    ejecutar()
    
def inicializarPoblacion():
    poblacion = []

    for i in range(9):
        solucion = np.random.randint(10, size=4)
        fitness, nn = evaluarFitness(solucion)
        print(fitness)
        poblacion.append(Nodo(solucion, fitness))

    return poblacion

def verificarCriterio(poblacion, generacion):
    if generacion == 15:
        return True

    return None

def evaluarFitness(solucion):
    valorFitness = 0

    # print(solucion)
    # print("alpha=" + str(hiperparametros[solucion[0]][0]) + ", lambd=" + str(hiperparametros[solucion[1]][1]) + ", iterations=" + str(hiperparametros[solucion[2]][2]) + ", keep_prob=" + str(hiperparametros[solucion[3]][3]) + "")

    nn = NN_Model(train_set, capas, alpha=hiperparametros[solucion[0]][0], lambd=hiperparametros[solucion[1]][1], iterations=hiperparametros[solucion[2]][2], keep_prob=hiperparametros[solucion[3]][3])
    nn.training(False)
    # Plotter.show_Model([nn])

    # print('Entrenamiento Modelo 1')
    # nn.predict(train_set)
    # print('Validacion Modelo 1')
    exactitud = nn.predict(val_set)
    # print('Pruebas Modelo 1')
    # nn.predict(test_set)
    # global modelos
    # modelos.append(nn)

    valorFitness = exactitud
    return valorFitness, nn

def seleccionarPadres(poblacion):
    poblacion = sorted(poblacion, key=lambda item: item.fitness, reverse=True)
    return [poblacion[0], poblacion[1], poblacion[2], poblacion[3], poblacion[4], poblacion[5]]
    # return [poblacion[0], poblacion[1]]

def emparejar(padres):
    nuevaPoblacion = padres

    for i in [0,2,4]:
    # for i in range(4):
        hijo = Nodo()
        # hijo.solucion = cruzar(padres[0], padres[1])
        hijo.solucion = cruzar(padres[i], padres[i + 1])
        hijo.solucion = mutar(hijo.solucion)
        hijo.fitness = evaluarFitness(hijo.solucion)
        nuevaPoblacion.append(hijo)

    return nuevaPoblacion

def cruzar(padre1, padre2):
    hijo = []
    for i in range(4):
        if np.random.uniform(0, 1) <= 0.5:
            hijo.append(padre1.solucion[i])
        else:
            hijo.append(padre2.solucion[i])    
    return np.array(hijo)

def mutar(solucion):
    solucion[np.random.randint(4)] = np.random.randint(10)
    return solucion

def imprimirPoblacion(poblacion):
    for individuo in poblacion:
        print('Individuo: ', individuo.solucion, ' Fitness: ', individuo.fitness, "alpha=" + str(hiperparametros[individuo.solucion[0]][0]) + ", lambd=" + str(hiperparametros[individuo.solucion[1]][1]) + ", iterations=" + str(hiperparametros[individuo.solucion[2]][2]) + ", keep_prob=" + str(hiperparametros[individuo.solucion[3]][3]) + "")

def imprimirMejorSolucion(poblacion, generacion):
    poblacion = sorted(poblacion, key=lambda item: item.fitness, reverse=True)
    print('\nGeneración: ', generacion, '\nMejor Solución: ', poblacion[0].solucion, '\nMejor Fitness: ', poblacion[0].fitness, '\n')
    print("alpha=" + str(hiperparametros[poblacion[0].solucion[0]][0]) + ", lambd=" + str(hiperparametros[poblacion[0].solucion[1]][1]) + ", iterations=" + str(hiperparametros[poblacion[0].solucion[2]][2]) + ", keep_prob=" + str(hiperparametros[poblacion[0].solucion[3]][3]) + "")
    global modelo_nn
    modelo_nn = poblacion[0].nn

def ejecutar():
    generacion = 0
    poblacion = inicializarPoblacion()
    fin = verificarCriterio(poblacion, generacion)

    while(fin == None):
        print('*************** GENERACION ', generacion, " ***************")
        imprimirPoblacion(poblacion)
        # Plotter.show_Model(modelos)
        # modelos = []
        padres = seleccionarPadres(poblacion)
        poblacion = emparejar(padres)
        generacion += 1
        
        fin = verificarCriterio(poblacion, generacion)
        
    print('*************** GENERACION ', generacion, " ***************")
    imprimirPoblacion(poblacion)
    imprimirMejorSolucion(poblacion, generacion)


if __name__ == "__main__":
    app.run(debug=True)
