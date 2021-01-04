import csv
import math
import numpy as np

municipios = []

def cargarDatos():
    cargarMunicipios()
    edades = []
    anios_i = []
    dist_u = []
    dataset = []
    with open('datasets/Dataset.csv', 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            if index > 0:
                data = []
                if(row[0] == 'Traslado'):
                    data.append(0)
                else:
                    data.append(1)
                if(row[1] == 'FEMENINO'):
                    data.append(0)
                else:
                    data.append(1)
                data.append(float(row[2]))
                data.append(calcularDistancia(row[3], row[5], row[6]))
                data.append(float(row[7]))
                edades.append(data[2])
                dist_u.append(data[3])
                anios_i.append(data[4])
                dataset.append(data)
            index += 1
        dataset = escalarVariables(dataset, [max(edades), min(edades)], [max(dist_u), min(dist_u)], [max(anios_i), min(anios_i)])

        result = np.array(dataset)
        #np.random.shuffle(result)
        result = result.astype(float).T
        # Se separa el conjunto de pruebas del de entrenamiento y de validacion
        slice_point_t = int(result.shape[1] * 0.7)
        # slice_point_v = int(result.shape[1] * 0.85)
        train_set = result[:, 0: slice_point_t].T
        # val_set = result[:, slice_point_t: slice_point_v].T
        # test_set = result[:, slice_point_v:].T
        val_set = result[:, slice_point_t:].T

        train_set_x = train_set[0:,1:]
        train_set_y = train_set[0:,0:1]
        val_set_x = val_set[0:,1:] 
        val_set_y = val_set[0:,0:1]
        # test_set_x = test_set[0:,1:]
        # test_set_y = test_set[0:,0:1]

        # print(result.shape[1])
        # print(slice_point_t)
        # print(slice_point_v)
        # print(train_set.shape)
        # print(val_set.shape)
        # print(test_set.shape)
        #print(train_set)
        #print(train_set_x.T.shape)
        # print(train_set_x.T.tolist())
        #print(train_set_y.T.shape)
        #print(train_set_y.T.tolist())
        #print(train_set_y.shape)

        # print(train_set_x.T.shape)
        # print(train_set_y.T.shape)
        # print(val_set_x.T.shape)
        # print(val_set_y.T.shape)
        # print(test_set_x.T.shape)
        # print(test_set_y.T.shape)

    return train_set_x.T, train_set_y.T, val_set_x.T, val_set_y.T#,  test_set_x.T, test_set_y.T

def cargarMunicipios():
    with open('datasets/Municipios.csv', 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            if index > 0:
                municipios.append(row)
            index += 1
    return

def calcularDistancia(cod_depto, cod_muni, nombre):
    for municipio in municipios:
        if(municipio[0] == cod_depto):
            if(municipio[1] == cod_muni):
                # ccordenada universidad = (14.589246, -90.551449)
                return haversine(14.589246, -90.551449, float(municipio[3]), float(municipio[4]))    

    print('no encontrado => ' + nombre)
    return 0

def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia

def escalarVariables(data, edad, dist, anio):
    for val in data:
        val[2] = (val[2]-edad[1])/(edad[0]-edad[1])
        val[3] = (val[3]-dist[1])/(dist[0]-dist[1])
        val[4] = (val[4]-anio[1])/(anio[0]-anio[1])

    return data