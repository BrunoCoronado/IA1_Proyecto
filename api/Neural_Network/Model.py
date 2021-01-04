import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos


class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)
        self.len_layers = len(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        #print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))
            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)
            self.actualizar_parametros(gradientes)
            if i % 10 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    def propagacion_adelante(self, dataSet):
        pesos = []
        for l in range(1, self.len_layers):
            pesos.append([self.parametros["W"+str(l)], self.parametros["b"+str(l)]])

        tmp = []
        c_anterior = dataSet.x # Se extraen las entradas
        predict = None
        for l in range(1, self.len_layers):
            Z_l = np.dot(pesos[l - 1][0], c_anterior) + pesos[l - 1][1]
            activacion = 'sigmoide'
            if ( l + 1 ) < self.len_layers:
                activacion = 'relu'
            A_l = self.activation_function(activacion, Z_l)
            c_anterior = A_l
            if ( l + 1 ) < self.len_layers:
                #Se aplica el Dropout Invertido
                D_l = np.random.rand(A_l.shape[0], A_l.shape[1]) #Se generan número aleatorios para cada neurona
                D_l = (D_l < self.kp).astype(int) #Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
                A_l *= D_l
                A_l /= self.kp    
                tmp.append([Z_l, A_l, D_l])
            else:
                predict = A_l
                tmp.append([Z_l, A_l])
        return predict, tmp

    def propagacion_atras(self, temp):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x

        pesos = []
        for l in range(1, self.len_layers):
            pesos.append(self.parametros["W"+str(l)])

        # (A1, D1, A2, D2, A3) = temp

        l = self.len_layers - 2
        # print('len(temp): ' + str(len(temp)))
        # print('l: ' + str(l))
        derivadas = []
        gradientes = {}
        while l > -1:
            if self.len_layers - 2 == l:
                #dZ3 = A3 - Y
                dZ_l = temp[l][1] - Y
                # dW3 = (1 / m) * np.dot(dZ3, A2.T) + (self.lambd / m) * W3
                dW_l = (1 / m) * np.dot(dZ_l, temp[l - 1][1].T) + (self.lambd / m) * pesos[l]
                # db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)    
                db_l = (1 / m) * np.sum(dZ_l, axis=1, keepdims=True)    

                derivadas.append([dZ_l, dW_l, db_l])
                gradientes["dZ" + str(l + 1)] = dZ_l
                gradientes["dW" + str(l + 1)] = dW_l
                gradientes["db" + str(l + 1)] = db_l
            else:
                # dA2 = np.dot(W3.T, dZ3)
                # print('len(derivadas):' +  str(len(derivadas)))
                dA_l = np.dot(pesos[l + 1].T, derivadas[len(derivadas) - 1][0])
                # dA2 *= D2
                dA_l *= temp[l][2]
                # dA2 /= self.kp        
                dA_l /= self.kp    
                # dZ2 = np.multiply(dA2, np.int64(A2 > 0))
                dZ_l = np.multiply(dA_l, np.int64(temp[l][1] > 0))
                # dW2 = 1. / m * np.dot(dZ2, A1.T) + (self.lambd / m) * W2
                if l == 0:
                    dW_l = 1. / m * np.dot(dZ_l, X.T) + (self.lambd / m) * pesos[l]
                else:
                    dW_l = 1. / m * np.dot(dZ_l, temp[l - 1][1].T) + (self.lambd / m) * pesos[l]
                # db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)    
                db_l = 1. / m * np.sum(dZ_l, axis=1, keepdims=True)    
                derivadas.append([dZ_l, dW_l, db_l, dA_l])
                gradientes["dZ" + str(l + 1)] = dZ_l
                gradientes["dW" + str(l + 1)] = dW_l
                gradientes["db" + str(l + 1)] = db_l
                gradientes["dA" + str(l + 1)] = dA_l

            l -= 1
        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet, mostrar=False):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        if mostrar:
            print("Exactitud: " + str(exactitud))
        return exactitud

    def predecir(self, dataSet, mostrar=False):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        return y_hat


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result