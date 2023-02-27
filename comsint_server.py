from flask import Flask, request, jsonify
from urllib.parse import unquote
from recomendaciones_comsint import Recomendador
import argparse

app = Flask(__name__)

# crea un objeto ArgumentParser
parser = argparse.ArgumentParser(description='Script de Flask para recomendaciones.')

# agrega un argumento para el archivo de recetas
parser.add_argument('-recetario', dest='recetario', required=False,
                    help='nombre del archivo CSV de datos con recetas')

# agregar un argumento para el máximo de registros que arroja el primer filtrado de similitud
parser.add_argument('-max_recetas', dest='max_recetas', required=False,
                    help='Máximo de registros que arroja el filtrado de similitud con canasta básica (default = todos)')

# agregar un argumento para el tamaño de los embeddings
parser.add_argument('-emb_size', dest='emb_size', required=False,
                    help='Tamaño de embeddings del modelo (default=128)')


# parsea los argumentos de la línea de comandos
args = parser.parse_args()

recetario = args.recetario
max_recetas = int(args.max_recetas)
emb_size = int(args.emb_size)

if recetario == None: recetario = 'recetario_mexicano_small.csv'
if max_recetas == None: max_recetas = -1
if emb_size == None: emb_size = 128

print('Recetario: ', recetario)
print('Máximo de recetas por petición:', max_recetas)
print('Embeddings size:', emb_size)

basedir = ''
Agente = Recomendador(basedir = basedir, fuente=recetario)
Agente.CargarModelo(emb_size=emb_size, version=4)

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

    # Filtrar recetas con los parámetros recibidos
    try:
        Agente.FiltrarRecetario_por_CanastaBasica(lista_ingredientes=ingredientes, 
                                                  similitud=similitud, max_rows=max_recetas, 
                                                  verbose=False)
        
        print(Agente.DF_RecetasFiltradas)
        print('Recetas en caché:', len(Agente.CACHE))

        df_filtrado = Agente.Calcular_InfoNutricional(verbose=True)
        df_filtrado = df_filtrado[df_filtrado['puntaje_platillo']>=puntaje_nutricion]
        df_filtrado = Agente.Calcular_Precios(df_filtrado, verbose=True)
        df_filtrado = df_filtrado[df_filtrado['costo_receta']<=max_precio].sort_values(by=['costo_receta','kcal', 'similitud'], ascending=True)
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



