from flask import Flask, request, jsonify
from urllib.parse import unquote
from recomendaciones_comsint import Recomendador
import argparse

app = Flask(__name__)

# crea un objeto ArgumentParser
parser = argparse.ArgumentParser(description='Script de Flask para recomendaciones.')

# agrega un argumento para el archivo
parser.add_argument('-recetario', dest='recetario', required=False,
                    help='nombre del archivo CSV de datos con recetas')

# parsea los argumentos de la línea de comandos
args = parser.parse_args()

# usa el valor del argumento del archivo
recetario = args.recetario

if recetario == None: recetario = 'recetario_mexicano_small.csv'

print('Recetario: ', recetario)

@app.route("/")
def index():
    return 'Servidor COMSINT recomendaciones está corriendo...'

@app.route("/prueba", methods=['GET'])
def prueba():    
    args = request.args

    for arg in args:
        print(arg, '=', args[arg])


    return "Que onda!"

@app.route('/filtrar_recetas', methods=['GET', 'POST'])
def filtrar_recetas():
    print('Filtrar recetas..\n')

    # Obtener los argumentos del url
    form = request.args
    request_correcto = True
    try:
        ingredientes = str(form['ingredientes'])
        ingredientes = unquote(ingredientes).strip()        
        similitud = float(form['similitud'])
        puntaje_nutricion = float(form['puntaje'])
        max_precio = float(form['precio'])
    except:
        print('ERROR: Las variables del url son incorrectas')
        print('Variables:')
        print('ingredientes, similitud, puntaje, precio')
        print('\nEjemplo: /filtrar_recetas?ingredientes=tomate,naranja,pepino&similitud=0.5&puntaje=0.7&precio=100\n')
        request_correcto = False
        pass

    if not request_correcto:       
        return 'Falló el request'

    basedir = ''
    Agente = Recomendador(basedir = basedir, fuente=recetario)
    Agente.CargarModelo(emb_size=128, version=4)

    # Filtrar recetas con los parámetros recibidos
    try:
        Agente.FiltrarRecetario_por_CanastaBasica(lista_ingredientes=ingredientes, similitud=similitud, max_rows=100, verbose=False)
        df_filtrado = Agente.Calcular_InfoNutricional(verbose=False)
        df_filtrado = df_filtrado[df_filtrado['puntaje_platillo']>=puntaje_nutricion]
        df_filtrado = Agente.Calcular_Precios(df_filtrado, verbose=False)
        df_filtrado = df_filtrado[df_filtrado['costo_receta']<=max_precio].sort_values(by=['costo_receta','similitud','kcal'], ascending=True)
        resultados = df_filtrado.to_dict(orient='records')
    except:
        print('No se encontraron recetas\n')
        resultados = []
   
    print(len(resultados), 'recetas encontradas.\n')
    

    return jsonify(resultados)

# Correr el servidor
if __name__ == '__main__':
    print('Corriendo el servidor de recomendaciones...')
    app.run(host='127.0.0.1', port=80)



