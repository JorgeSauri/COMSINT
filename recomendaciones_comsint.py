# Requerimientos de librerías:
# pip install spacy
# python -m spacy download es_core_news_md
# !pip install transformers
# !pip install transformers scipy ftfy accelerate


# Importar librerías
import pandas as pd
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

from transformers import TFDistilBertModel, DistilBertTokenizerFast
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, Conv1D, MaxPool1D, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import config
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ReduceLROnPlateau
import os.path

class Recomendador():
    ############################################################################################################
    ############################################################################################################
    ####################################### SISTEMA DE RECOMENDACIONES #########################################    
    ################################ UNIVERSIDAD INTERNACIONAL DE LA RIOJA #####################################
    ############################################################################################################
    ########################################## JORGE SAURI CREUS ###############################################    
    ######################################### RUBI GUTIERREZ LOPEZ #############################################
    ############################################################################################################

    NUM_RECETAS = 5000
    EMB_SIZE = 128
    VOCAB_SIZE = 768
    INFO_COLS = ['kcal','carbohydrate', 'protein', 'total_fat']
    basedir = ''

    # Calificar el platillo de acuerdo a:
    # Carb% = (45%-65%)  
    # Prot% = (10%-35%)   
    # Grasa% = (20%-35%)  
    # Fuente:
    # Manore, M.M. Exercise and the institute of medicine recommendations for nutrition. 
    # Curr Sports Med Rep 4, 193–198 (2005). 
    # https://link.springer.com/article/10.1007/s11932-005-0034-4 

    RANGO_CARBOHIDRATOS = range(45, 66) 
    RANGO_PROTEINAS = range(10, 36)
    RANGO_GRASAS = range(20, 36) 

    def __init__(self,
                 basedir = '',
                 fuente='recetas.csv',
                 nutricion='nutricion.csv',
                 canasta='canasta_basica.csv',
                 precios='lista_precios_profeco_2022.csv',
                 encoding="ISO-8859-1"):
        """
            La clase Recomendador carga los datasets: fuente, nutricion y canasta; e inicializa los parámetros
            para recomendar una lista de recetas de acuerdo a ciertas características. Utiliza la librería de DeepLearning
            SpaCy para NLP para buscar similitudes entre el recetario (fuente) y los ingredientes de la canasta básica (canasta).

            Una vez instanciada la clase, para poder utilizarla por primera vez, hay que llamar al método: ProcesarRecetario() para
            obtener una lista de elementos más similares a la canasta básica proporcionada, y luego se llama al método Calcular_InfoNutricional()
            si se desea agregarle a ésta lista la información nutricional y costos de cada receta (Esta información es estimada, se
            utiliza SpaCy para encontrar similitudes entre cada ingrediente de la canasta básica y cada ingrediente de la receta,
            sin embargo los valores tanto nutricionales como de costos dependerán mucho del dataframe proporcionado en el parámetro 'nutricion'.

            Parámetros:
            -----------------------------------------------------------------------------------------------------------
            @basedir: Directorio base o raíz donde está corriendo la clase. 
                      A partir de esta raíz, se utilizan los sub-directorios: datasets y modelos
            @fuente: Archivo csv del recetario. (La clase lo buscará en: basedir/datasets/#####.csv)
            @nutricion: Archivo csv del dataset de información nutricional. (La clase lo buscará en: basedir/datasets/#####.csv)
            @canasta: Archivo csv de la canasta básica. (La clase lo buscará en: basedir/datasets/#####.csv)
            @precios: Archivo csv con la lista de precios de profeco de todos los productos de alimentación. (La clase lo buscará en: basedir/datasets/#####.csv)
            @encoding: Tipo de codificación del archivo (utf-8 o iso-8859-1)
        """

        # Si hay hardware de GPU, inicializarlo y configurarlo para que administre bien la memoria del hardware
        try:
            physical_devices = config.list_physical_devices('GPU') 
            config.experimental.set_memory_growth(physical_devices[0], False)        
            tf.config.set_visible_devices([], 'GPU')
            print(physical_devices)
        except:
            print('No se encontró hardware GPU')
            pass

        self.basedir = basedir

        # cargamos el modelo entrenado en español
        self.nlp = spacy.load("es_core_news_md")

        # Diccionario de medidas más comunes en recetas
        self.Medidas = {     
            'libras': ['libra ', 'libras ', 'lb ', 'lb.'],           
            'onzas': ['onza ', 'onzas ', 'oz ', 'ozs ', 'onza.', 'onzas.', 'oz.', 'ozs.'],
            'kilos': ['kilo ', 'kilos ', 'kg ', 'k ', 'kgr ', 'kilo.', 'kilos.', 'kg.', 'k.', 'kgr.', 'kilogramos ', 'kilogramos.'],
            'miligramos': ['mg ', 'miligramo ', 'miligramos ', 'mgr ', 'mg.', 'mgr.', 'miligramos.', 'miligramo.'],
            'gramos': ['gramos ', 'gr ', 'g ', 'gram ', 'grams ', 'gr.', 'g.', 'gram.', 'grams.', 'gramos.'],
            'litros': ['litro ', 'litros ', 'l ', 'lt ', 'ltr ', 'ltrs ', 'l.', 'lt.', 'ltr.', 'ltrs.', 'litros.', 'litro.'],
            'mililitros': ['mililitro ', 'mililitros ', 'ml ', 'mltr ', 'mltrs ', 'ml.', 'mltr.', 'mltrs.'],        
            'piezas': ['pieza ', 'piezas ', 'unidad ', 'unidades ', 'pz ', 'pza ', 'pz.', 'pza.', 'pieza.', 'piezas.', 'unidad.', 'unidades.', 'pz.', 'pza.'],
            'tazas': ['taza ', 'tazas ', 'tza ', 'tz ', 'cup ', 'cups ', 'tza.', 'tz.', 'taza.', 'tazas.', 'tza.', 'tz.', 'cup.', 'cups.'],
            'cucharadas': ['cucharada ', 'cucharadas ', 'cuch ', 'cda ', 'cdas ', 'cuch.', 'cda.', 'cdas.', 'tbsp.','cucharada.', 'cucharadas.', 'cuch.', 'cda.', 'cdas.','tbsp.'],
            'cucharaditas': ['cucharadita ', 'cucharaditas ', 'cdta ', 'cdtas ', 'cdta.', 'cdtas.', 'tsp ', 'tsp.', 'cucharadita.', 'cucharaditas.']
            
        }

        self.stopwords = ["el", "para", "con", "en", ",", "contra",
                          "de", "del", "la", "las", "los", "un",
                          "una", "unos", "unas", "o", "ó", "y"]

        # Dataframes:
        self.DF_RecetasFiltradas = None

        if (fuente != ''): self.df_recetario = pd.read_csv(self.basedir + 'datasets/' + fuente, encoding=encoding)
        if (nutricion != ''): self.df_nutricion = pd.read_csv(self.basedir + 'datasets/' + nutricion, encoding=encoding)
        if (canasta != ''): self.df_canasta = pd.read_csv(self.basedir + 'datasets/' + canasta, encoding=encoding)
        if (precios != ''): self.df_precios = pd.read_csv(self.basedir + 'datasets/' + precios, encoding=encoding)

        

    ############################################################################################################
    ############################################################################################################
    ## UTILERÍAS AUXILIARES PARA LA CLASE
    ############################################################################################################
    ############################################################################################################
    def LimpiarString(self, cadena):
        """
        Limpia una cadena de caracteres extraños, simbolos, stopwords, y unidades de medida

        @cadena: String a limpiar

        Devuelve: La cadena limpia
        """
        result = []
        for c in list(cadena.lower()):
            if (c == ';' or c == '+' or c == '-'):
                c = ','
            else:
                if (not c.isalpha()):
                    c = ' '
            result.append(c)
        result = ''.join(result).split(' ')

        result2 = []
        for e in result:
            if (e != ''):
                result2.append(e)

        result = ''
        for e in result2:
            result += str(e) + ' '

        MedidasList = []
        for medida in self.Medidas:
            for abr in self.Medidas[medida]:
                MedidasList.append(abr.strip())

        # Eliminar unidades de medida que se colaron en los ingredientes
        result = ' '.join([medida for medida in result.split(' ') if medida not in MedidasList])

        # Eliminar las stopwords comunes
        result = ' '.join([word for word in result.split(' ') if word not in self.stopwords])

        return str(result)

    def encontrar_unidades(self, cadena):
        """
        Procesa un string que presuntamente contiene la cantidad y unidad de un ingrediente
        y devuelve en que unidad se está midiendo

        @cadena: El string con la cantidad, unidad e ingrediente (Ej. '25 gramos de harina de maíz')

        Devuelve: La unidad de medida
        """

        cadena = cadena.lower()

        # Por defecto regresaremos la unidad 'pieza'
        result = 'piezas'
        cantidad = 1
        
        sin_cantidad = False
        acabar = False

        for medida in self.Medidas:
            for abr in self.Medidas[medida]:
                index = cadena.find(abr)                               
                if index > -1 and not acabar:                                                                             
                    cantidad = cadena.split(abr)[0].strip()                                                       
                    for subcad in cantidad.split(' '):
                        if subcad.isnumeric:
                            try:
                                cantidad = float(subcad.strip())                                                              
                                sin_cantidad = False
                            except:                               
                                sin_cantidad = True
                    if sin_cantidad:
                        cantidad = 1                         
                    result = medida
                    acabar = True
                    break                
                        
        return cantidad, result

    def convertir_a_gramos(self, cantidad, unidad):
        Convercion = {
                'libras': 453.6,                
                'onzas': 28.35,
                'kilos': 1000,
                'miligramos': 0.0009,
                'gramos': 1,
                'litros': 1000,
                'mililitros': 0.0009,        
                'piezas': 100,
                'tazas': 0.213,  
                'cucharadas': 0.0135,
                'cucharaditas': 0.0045
                }    
        result = 0.0
        # encontrar que unidad del diccionario es:
        for medida in self.Medidas:
            if unidad == medida:
                factor = Convercion[unidad]
                result = cantidad * factor
                break
        return round(result, 4)
 
    def separar_ingredientes_spacy(self, cadena):
        """
          Recibe un string con los ingredientes mezclados y separados por coma con cantidades, unidades y descripciones,
          la procesa, y devuelve 3 listas con los ingredientes, sus cantidades y sus unidades de medida por separado

          Parámetros:
          -----------------------------------------------------------------------------------------------------------
          @cadena: String con todos los ingredientes, cantidades y unidades como viene en la receta
          -----------------------------------------------------------------------------------------------------------

          Devuelve:
          3 listas con los ingredientes separados individualmente:
          cantidades, unidades, ingredientes_texto

        """

        # Inicializa las listas para las cantidades, las unidades y los ingredientes
        cantidades = []
        unidades = []
        ingredientes_texto = []

        for cad in cadena.split(','):
            # Procesa la cadena como un documento de spaCy
            doc = self.nlp(cad)

            cantidad = 0
            unidad = None
            ingrediente_texto = ''

            # Recorre cada token en el documento
            for token in doc:
                # Si el token es un número, lo agrega a la lista de cantidades
                if token.like_num and token.text.isnumeric():
                    cantidad = float(token.text)
                    # Buscar la unidad
                    unidad = self.encontrar_unidades(cad.split(token.text)[1])[0]
                    ingrediente_texto = cad.split(token.text)[1]
                    ingrediente_texto = self.LimpiarString(ingrediente_texto)

                    # Agrega la cantidad, la unidad y el ingrediente a las listas
                    cantidades.append(cantidad)
                    unidades.append(unidad)
                    ingredientes_texto.append(ingrediente_texto)
                    break
                else:
                    # Analizar si son fracciones en ASCII: '¼', '½', '¾'
                    # chr(188), chr(189), chr(190)
                    CharFracc = ['¼', '½', '¾', '1/4', '1/2', '1/3', '3/4']
                    NumFracc = [1 / 4, 1 / 2, 3 / 4, 1 / 4, 1 / 2, 1 / 3, 3 / 4]
                    for i in range(len(CharFracc)):
                        cf = CharFracc[i]
                        nf = NumFracc[i]
                        if token.text.strip() == cf:
                            cantidades.append(nf)
                            # Buscar unidades
                            unidad = self.encontrar_unidades(cad.split(token.text)[1])[0]
                            unidades.append(unidad)
                            ingrediente_texto = cad.split(token.text)[1]
                            ingrediente_texto = self.LimpiarString(ingrediente_texto)
                            ingredientes_texto.append(ingrediente_texto)
                            break

                            # Devuelve las listas
        return cantidades, unidades, ingredientes_texto





    ############################################################################################################
    ############################################################################################################
    ## FILTRADO DE RECETAS CON BASE EN LA CANASTA BÁSICA:
    ############################################################################################################
    ############################################################################################################

    def FiltrarRecetario_por_CanastaBasica(self,
                          col_title='nombre_del_platillo', col_ingredientes='ingredientes',
                          similitud=0.6, max_rows=20, verbose=True):
        """
          Procesa el recetario cargado al instanciar la clase, y trata de encontrar las recetas más
          similares en cuanto a lista de ingredientes con el dataset de canasta básica.

          Parámetros:
          -----------------------------------------------------------------------------------------------------------
          @col_title: Nombre de la columna del csv del titulo de la receta
          @col_ingredientes: Nombre de la columna del csv con los ingredientes
          @similitud: Similitud de ingredientes mínima permitida con la lista de la canasta básica
          @max_rows: Número máximo de filas que devuelve la función ordenadas de mayor a menor similitud
          @verbose: Indica si se imprimen mensajes durante el proceso
          -----------------------------------------------------------------------------------------------------------

          Devuelve:
          Dataframe de pandas con las columnas: 'platillo', 'ingredientes', 'similitud'

        """

        # Limpiar el dataframe de recetas
        canasta = ','.join([prod for prod in self.df_canasta['producto']])

        # Limpiar el dataframe de información nutricional
        self.df_nutricion['nombre'] = self.df_nutricion['nombre'].str.lower()

        Platillos = []
        Ingredientes = []
        Sim = []

        print('Buscando recetas con ingredientes de la canasta básica... \n')
        for i in tqdm(range(len(self.df_recetario))):
            row = self.df_recetario.iloc[i]
            ingredientes_clean = self.LimpiarString(row[col_ingredientes])
            tokenIngredientes = self.nlp(ingredientes_clean)
            similaridad = tokenIngredientes.similarity(self.nlp(canasta))
            if similaridad > similitud:
                Platillos.append(row[col_title])
                Ingredientes.append(row[col_ingredientes])
                Sim.append(round(similaridad,1))

        dfFiltrados = pd.DataFrame(list(zip(Platillos, Ingredientes, Sim)),
                                   columns=['nombre_del_platillo', 'ingredientes', 'similitud'])

        dfFiltrados = dfFiltrados.sort_values(by=['similitud'], ascending=False)[:max_rows]

        print(' \n\n', len(dfFiltrados), 'platillos encontrados con similitud mayor a', similitud)

        # Guardamos el dataframe en una variable de la clase, y también la regresamos
        self.DF_RecetasFiltradas = dfFiltrados

        if (verbose): return dfFiltrados


    ############################################################################################################
    ############################################################################################################
    ## MÉTODOS PARA EL CÁLCULO DE INFORMACIÓN NUTRICIONAL:
    ############################################################################################################
    ############################################################################################################

    def feature_vector_similarity(self, feature_vec1, feature_vec2):        
        """
        Calcula la similitud entre dos vectores de características utilizando la distancia coseno

        @feature_vec1: Vector de características
        @feature_vec2: Vector de características

        Regresa:
        Un float entre 0 y 1 que inidica la similitud entre los 2 vectores proporcionados
        """         
        cos_sim = tf.keras.losses.cosine_similarity(feature_vec1, feature_vec2)
        return cos_sim

    def get_feature_vectors(self, ingredient_list, max_len=128):
        """
        Utiliza DistilBERT para obtener los vectores de características de una lista de ingredientes.
        Primero tokeniza el string de que recibe de entrada utilizando un BERT Tokenizer, 
        luego generamos un tensor de entrada con padding para que tenga las mismas dimensiones,
        y por último le pasamos estos tokens al modelo BERT pre-entrenado para que nos devuelva una
        matriz de embbedings de shape: (max_len, 768), que es nuestro array de características que
        representa los tokens de la entrada inicial en el espacio de embbedings del modelo pre-entrenado.

        Nota: Para convertirlo a vector es necesario hacerle un flatten() a la salida de esta función.

        @ingredient_list: Lista de ingredientes en formato string separado por comas.
        @max_len: Número de dimensiones de la matriz de embbedings (default=64, mejor=512).

        Regresa: Un vector NumPy de shape (max_len, 768)
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")   
        model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=False)   
        input_ids = tf.constant(tokenizer.encode(ingredient_list, return_tensors='tf'))    
        input_ids = pad_sequences(input_ids, maxlen=max_len, padding='post')
        pooled_output = model(input_ids)[0]  

        feature_vec = pooled_output.numpy().squeeze()
        return feature_vec      

    def generar_dataset_entrenamiento_nut(self, 
                                    df_nutricionales = '',                                   
                                    encoding='ISO-8859-1',
                                    usecols=['nombre', 'kcal','carbohydrate', 'protein', 'total_fat'],
                                    min_ingredientes = 3,
                                    max_ingredientes = 10, 
                                    min_unidades = 5,
                                    max_unidades = 20,
                                    min_kcal = 0,
                                    max_kcal = 10000,
                                    healthy_only = False,
                                    numero_recetas=100):
        """
        Regresa un NumPy Array para entrenar un modelo de regresión.
        Por defecto se toman las columnas: 'nombre', 'kcal','carbohydrate', 'protein', 'total_fat'
        Que son las columnas del dataframe de nutricion que usamos para entrenar.
        El método toma al azar de min_ingredientes a max_ingredientes y genera también cantidades en gramos aleatorias.
        Con esta información el método genera recetas ficticias con su correcto contenido energético y nutricional.
        Este dataset ficticio puede usarse para entrenar modelos de regresión para el cálculo energético.

        Parámetros:
        @df_nutricionales: El dataframe de donde se toman la información nutricional
        @encoding: El formato de encoding del archivo csv, por ejemplo: UTF-8 o ISO-8859-1
        @usecols: Los nombres de las columnas del csv que se codificarán en el array
        @min_ingredientes: El número mínimo de ingredientes que puede tener una receta
        @max_ingredientes: El máximo número de ingredientes que puede tener una receta

        Devuelve:
        - Un NumPy Array con dtype=string (Antes de usarlo, es necesario convertir los valores numéricos a float16 o float32 etc.) 
        

        Ejemplo:
            dataset = generar_dataset_entrenamiento_nut(numero_recetas=1000, min_ingredientes=5, max_ingredientes=10)
        """

        if df_nutricionales == '':
            df = self.df_nutricion
        else:
            df = pd.read_csv(self.basedir + 'datasets/' + df_nutricionales, encoding=encoding, usecols=usecols)
        
        
        print('Generando', numero_recetas,' recetas aleatorias...\n')
        
        RecetaRandom = []

        lista_medidas_chicas = ['onzas',
                                'gramos',
                                'mililitros',
                                'piezas',
                                'tazas',
                                'cucharadas',
                                'cucharaditas']
        

        for i_recetas in tqdm(range(numero_recetas)):
                nombre = ''
                kcal = 0.0
                gramos_carb = 0.0
                gramos_proteina = 0.0
                gramos_grasa = 0.0

                agregar_receta = False
                while (not agregar_receta):
                    agregar_receta = False
                    check_kcal = False
                    check_saludable = False

                    for i_ingredientes in range(np.random.randint(min_ingredientes, max_ingredientes+1)):
                        # Elegir un ingrediente al azar el dataframe de nutricionales
                        i_rand = np.random.randint(len(df))
                        cant_rand = np.random.randint(min_unidades,max_unidades)

                        unidades = lista_medidas_chicas[np.random.randint(len(lista_medidas_chicas))]

                        cant_rand_gr = self.convertir_a_gramos(cant_rand, unidades)

                        row_alimento = df.iloc[i_rand]
                        nombre += str(cant_rand) + ' ' + unidades +' de ' + str(row_alimento['nombre']).lower().replace(',', ' ').strip() + ', '
                        # Como el dataset de nutrición viene en porciones de 100g cada medida
                        kcal += cant_rand_gr * (float(str(row_alimento['kcal']))/100)       
                        gramos_carb += cant_rand_gr * (float(str(row_alimento['carbohydrate']).replace(' ', '').split('g')[0]) / 100)
                        gramos_proteina += cant_rand_gr * (float(str(row_alimento['protein']).replace(' ', '').split('g')[0]) / 100)                               
                        gramos_grasa += cant_rand_gr * (float(str(row_alimento['total_fat']).replace(' ', '').split('g')[0]) / 100)          

                    nombre = nombre[:-2]

                    # Tope de kcals:
                    if (kcal in range(min_kcal, max_kcal+1)): check_kcal = True
                    
                    # Si healty_only es True, solo acepta recetas en los rangos saludables
                    if healthy_only:
                        check_saludable = (gramos_carb in self.RANGO_CARBOHIDRATOS 
                                            and gramos_proteina in self.RANGO_PROTEINAS 
                                            and gramos_grasa in self.RANGO_GRASAS)      
                    else:
                        check_saludable = True                  

                    if check_kcal and check_saludable:
                        agregar_receta = True
                        RecetaRandom.append([nombre, round(kcal,2), 
                                            round(gramos_carb,2), 
                                            round(gramos_proteina,2), 
                                            round(gramos_grasa,2)]
                                            )
                
                
        result = np.array(RecetaRandom)


        return result


    def procesar_dataset_validacion(self, 
                                df_recetas, 
                                encoding='ISO-8859-1',
                                col_nombre_receta = 'nombre_del_platillo',
                                col_nombre_porcion = 'serving_size',
                                col_nombre_ingredientes = 'ingredientes',                               
                                usecols=['kcal','carbohydrate', 'protein', 'total_fat']):
        """
        Carga un dataframe con recetas y su información nutricional, y regresa un NumPy Array para entrenar un modelo de regresión.
        Por defecto se toman las columnas: 'nombre', 'kcal','carbohydrate', 'protein', 'total_fat'
        Que son las columnas del dataframe de nutricion que usamos para entrenar.
        

        Parámetros:
        @df_recetas: El dataframe de donde se toma la información nutricional
        @encoding: El formato de encoding del archivo csv, por ejemplo: UTF-8 o ISO-8859-1
        @usecols: Los nombres de las columnas del csv que se codificarán en el array

        Devuelve:
        - Un NumPy Array con dtype=string (Antes de usarlo, es necesario convertir los valores numéricos a float16 o float32 etc.) 
        
        """

        df = pd.read_csv(self.basedir + 'datasets/' + df_recetas, encoding=encoding)
        print('Procesando', df_recetas)
        Receta = []

        for i_recetas in tqdm(range(len(df))):
                row = df.iloc[i_recetas]               
                platillo = str(row[col_nombre_receta])
                ingredientes = str(row[col_nombre_ingredientes])
                serving_size = str(row[col_nombre_porcion])
                
                kcal = float(str(row[usecols[0]]))
                
                #carbs
                try:
                    gramos_carb = float(str(row[usecols[1]]))
                except:
                    cads = str(row[usecols[1]]).split('gr')                   
                    gramos_carb = 0.0
                    for c in cads:
                        if c.strip().isnumeric: 
                            gramos_carb = float(c)
                            break                
                #proteinas
                try:
                    gramos_proteina = float(str(row[usecols[2]]))
                except:
                    cads = str(row[usecols[2]]).split('gr')                   
                    gramos_proteina = 0.0
                    for c in cads:
                        if c.strip().isnumeric: 
                            gramos_proteina = float(c)
                            break      
                #grasas
                try:
                    gramos_grasa = float(str(row[usecols[3]]))
                except:
                    cads = str(row[usecols[3]]).split('gr')                   
                    gramos_grasa = 0.0
                    for c in cads:
                        if c.strip().isnumeric: 
                            gramos_grasa = float(c)
                            break                  
               
                Receta.append([ingredientes, round(kcal,2), round(gramos_carb,2), round(gramos_proteina,2), 
                                    round(gramos_grasa,2)])
                
                
        result = np.array(Receta)


        return result        



    def calcular_feature_vecs(self, array_recetas, max_len=128, 
                                    save=True, verbose=True, 
                                    sufix='_recetas_random'):

        """
        Método que recibe un array de recetas con el siguiente formato:
        - La primera columna del array debe ser un string con los ingredientes y sus cantidades:
            Formato ej: 10gr Manzana, 4.5gr azúcar, etc.
        - El resto de las columnas son valores de contenido energético (deben ser numéricos)
        
        La función recorre el arreglo, tokenizando y calculando el vector de características de 
        la primera columna de texto, utilizando el TDistilBERTTokenizer de la función get_feature_vectors().

        Con este arreglo puede entrenarse una red neuronal que aprenda a inferir los valores energéticos
        tomando como entrada una matriz de embbedings de TDistilBERT.

        Parámetros:
        @array_recetas: Un arreglo con las recetas y su información nutricional en formato numPy array.
        @max_len: El número máximo de tokens para la matriz de embedings. 
        @save: Indica si se guardan los arrays numpy
        @verbose: Muestra los mensajes durante el proceso

        Devuelve 2 arreglos:
        x: un arreglo con todas las matrices de embbedings para usar como entrada a un modelo
        y: un arreglo con todos los vectores de información nutricional, uno por cada matriz de embbedings.

        Ejemplo:
        dataX, dataY = calcular_feature_vecs(dataset_entrenamiento, max_len=128)
        """
        
        # Recorremos cada receta del array para calcular su similitud con la lista de ingredientes

        result_x = []
        result_y = []

        if (verbose): print('Calculando vector de características de', len(array_recetas), 'recetas...')
        if (verbose): 
            for i in tqdm(range(len(array_recetas))):
                # Generando el data X con el vector caracteristicas del texto usando DistilBERT
                receta = array_recetas[i][0]                                     
                feature_vec_receta = self.get_feature_vectors(receta, max_len=max_len)                    
                feature_vec_receta = feature_vec_receta.flatten()
                result_x.append(feature_vec_receta)

                # Generando el data Y con el resto de las columnas
                result_y.append([float(array_recetas[i][val]) for val in range(1, array_recetas.shape[1])])
        else:
            for i in range(len(array_recetas)):
                # Generando el data X con el vector caracteristicas del texto usando DistilBERT
                receta = array_recetas[i][0]                                     
                feature_vec_receta = self.get_feature_vectors(receta, max_len=max_len)                    
                feature_vec_receta = feature_vec_receta.flatten()
                result_x.append(feature_vec_receta)

                # Generando el data Y con el resto de las columnas
                result_y.append([float(array_recetas[i][val]) for val in range(1, array_recetas.shape[1])])

        result_x = np.array(result_x, dtype=np.float16)
        result_y = np.array(result_y, dtype=np.float16)

        # Guardar los arrays a disco
        if save:
            np.save(self.basedir + 'datasets/numpy/' + str(len(array_recetas)) + sufix.strip() + '_EMBED-'+ str(max_len) +'_DATA_X', result_x)
            np.save(self.basedir + 'datasets/numpy/' + str(len(array_recetas)) + sufix.strip() + '_EMBED-'+ str(max_len) +'_DATA_Y', result_y)

        return result_x, result_y




    ############################################################################################################
    ############################################################################################################
    ## MODELO DE REGRESIÓN LINEAL PARA CÁLCULO DE INFORMACIÓN NUTRICIONAL
    ## TOMANDO COMO ENTRADA UN VECTOR DE CARACTERÍSTICAS GENERADO POR UN MODELO DE LENGUAJE.
    ############################################################################################################
    ############################################################################################################

    def GenerarModeloRegresionCNN(self, input_shape, emb_size, numero_salidas, kernels=128):
            """
            Devuelve un modelo de CNN 1D para aprender 
            los patrones de ingredientes y sus valores nutricionales.

            Parámetros:
            @input_shape: El shape de entrada o input shape del vector de características.
                          NOTA: El modelo recibe un VECTOR, y luego hace un reshaping al formato necesario
                                para las capas de convolución, no es necesario modificar el shape del vector.

            @emb_size: El tamaño de embbeding que se utilizó (el tamaño del vocabulario para DistilBERT es 768)
            @numero_salidas: El número de columnas o valores que aprenderá a predecir.
            @kernels: Un factor base para el número de kernels o filtros de las capas de convolución.
                      Toma como base el parámetro pasado 'kernels' y luego lo multiplica por 4 para aumentar el número de filtros.                     

            Devuelve: 
            Una instancia de la clase tensorflow.keras.Model

            Ejemplo:
            modelo = GenerarModelo(emb_size=512, numero_salidas=y_train.shape[1])
            modelo.compile(RMSprop(learning_rate=1e-5), loss="mean_absolute_error", metrics=['mae'])
            modelo.summary()

            history = modelo.fit(x = x_train, y = y_train,
                                batch_size = 8,
                                epochs = 30,
                                validation_data=[x_test, y_test])

            modelo.save('MiModelo.h5')

            """

            input_tensor = Input(shape=input_shape, name='CapaEntrada')

            # Para no cambiar el shape de los inputs, le hacemos un reshape antes de pasarlo a la CONV1D:
            reshaped = Reshape(input_shape=(-1,input_shape), target_shape=(emb_size, 768), name='RESHAPING')(input_tensor)

            # Capa de normalización de las entradas
            cnn = BatchNormalization()(reshaped)
            # Capas de convolución 1D
            cnn = Conv1D(kernels*4, 5, activation='relu', name='CONV_1')(cnn)       
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_1')(cnn)
            cnn = Conv1D(kernels*2, 3, activation='relu', name='CONV_2')(cnn)
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_2')(cnn)
            cnn = Conv1D(kernels, 3, activation='relu', name='CONV_3')(cnn)
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_3')(cnn)
            cnn = Dropout(0.2)(cnn)
            cnn = Flatten()(cnn)

            # Capas densamente conectadas para aprender características y patrones 
            x = Dense(256, activation='relu')(cnn)                    
            x = Dense(128, activation='relu')(x)              
            x = Dense(64, activation='relu')(x)             
            x = Dropout(0.25)(x)         

            # Capa de salida de regresión:        
            output_tensor = Dense(numero_salidas, activation='relu', name='CapaSalida')(x)

            # Construimos el modelo
            model = Model(inputs=input_tensor, outputs=output_tensor, name="ModeloCNNNut_"+str(kernels))
            model.build(input_shape)

            # Guardamos el modelo en una propiedad de la clase
            self.modeloCNN = model

            # Y también retornamos el modelo
            return model


    ############################################################################################################
    ############################################################################################################
    ## MÉTODO PARA MOSTRAR MÉTRICAS DE EVALUACIÓN DE NUESTROS MODELOS
    ############################################################################################################
    ############################################################################################################
    def EvaluarModeloRegresion(self, INFO_COLS, history, x_val, y_val, modelo=None):
        """
        Evaluar y graficar el entrenamiento de un modelo de regresion.

        Parámetros:
        @INFO_COLS: Un List con el nombre de las columnas o features        
        @history: Una instancia del history regresado por model.fit()
        @x_val: Un array con las entradas de validación
        @y_val: Un array con los y_true de validación
        @modelo: La instancia del modelo a evaluar (si es None, se utiliza el self.modeloCNN)

        Devuelve: None
        """
        try:
            if modelo==None:
                modelo = self.modeloCNN
        except:
            pass
        
        try:
            pd.DataFrame(history.history).plot()
            plt.show()
        except:
            pass

        scores = modelo.evaluate(x_val, y_val)
        print(scores)

        test_predictions = modelo.predict(x_val)
        sum_acc = []
        for i in range(len(y_val)):
                for j in range(len(INFO_COLS)):
                    if y_val[i][j] <= test_predictions[i][j]: 
                        if (test_predictions[i][j]>0):
                            acc = y_val[i][j] / test_predictions[i][j]
                        else:
                            acc = 0.0
                    else:
                        if (y_val[i][j]>0):
                            acc = test_predictions[i][j] / y_val[i][j]
                        else:
                            acc = 0.0

                    sum_acc.append(np.abs(acc))
                    print('receta',i, INFO_COLS[j]+'_true:', y_val[i][j], INFO_COLS[j]+'_pred:', 
                            test_predictions[i][j], ' precisión:', round(acc * 100, 1),'%')
                print('---------------------------------------------------------------------------')
        print('Precisión promedio aprox. = ', round(np.mean(sum_acc)*100,2),'%')
        return


    ############################################################################################################
    ############################################################################################################
    ## CARGA DE ARCHIVOS NUMPY GUARDADOS EN DISCO
    ############################################################################################################
    ############################################################################################################
    def CargarNumpyRecetas(self, NUM_RECETAS, EMB_SIZE, verbose=True, sufix='_recetas_random'):
        """
        Carga los arreglos X e Y desde archivos tipo npy (NumPy).
        Utiliza los parámetros para armar el nombre del archivo a buscar.

        Parámetros:
        @basedir: Directorio base donde el método intentará buscar los archivos 
        @NUM_RECETAS: El número de recetas que contienen los archivos NumPy.
        @EMB_SIZE: El tamaño del embbeding que se utilizó para generarlos.

        Devuelve:
        Una tupla de numpy arrays X, Y. Si están vacíos, no encontró los archivos.
        """
        x = np.array([])
        y = np.array([])

        archivoX = self.basedir + 'datasets/numpy/' + str(NUM_RECETAS) + sufix.strip() + '_EMBED-'+ str(EMB_SIZE) +'_DATA_X.npy'
        archivoY = self.basedir + 'datasets/numpy/' + str(NUM_RECETAS) + sufix.strip() + '_EMBED-'+ str(EMB_SIZE) +'_DATA_Y.npy'
        check_fileX = os.path.isfile(archivoX) 
        check_fileY = os.path.isfile(archivoY)

        if check_fileX and check_fileY:
            x = np.load(archivoX)
            if (verbose): print(archivoX, 'cargado con éxito.')
            y = np.load(archivoY)
            if (verbose): print(archivoY, 'cargado con éxito.')
        else:
            if (verbose): 
                print('Error al cargar archivos NumPy.')
                if not check_fileX: print(archivoX, 'no existe o está corrupto.')
                if not check_fileY: print(archivoY, 'no existe o está corrupto.')        
        return x, y


    ############################################################################################################
    ############################################################################################################
    ## MÉTODO PARA ENTRENAR NUESTRO MODELO, YA SEA GENERANDO DATASETS DE RECETAS FICTICIAS
    ## O BIEN, CARGANDO UN DATASET REAL DE RECETAS EN CSV Y PRE-PROCESÁNDOLO PARA ENTRENAR
    ## TAMBIÉN RECIBE PARÁMETROS ESPECIALES PARA LA GENERACIÓN ALEATORIA DE RECETAS.
    ############################################################################################################
    ############################################################################################################
    def EntrenarModelo(self, df_nutricionales='nutricion.csv', 
                            df_training='',
                            df_test='', df_val='',
                            min_ingredientes=5, max_ingredientes=15,
                            min_unidades=5, max_unidades=20,
                            learning_rate = 1e-4,
                            batch_size = 8,
                            initial_epoch=0,
                            epochs = 20,
                            version =4, kernels=128,                   
                            steps_per_epoch = None,                       
                            verbose=True, save=True, savenumpy=False):
        """
        Entrenar el modelo de cálculo de información nutricional

        Parámetros:
        @df_nutricionales: El dataset de valores nutricionales con el que se arma el dataset de entrenamiento
        @df_training: Si hay un dataset en csv para entrenar, lo utiliza en vez de generar recetas ficticias
        @min_ingredientes: Mínimo de ingredientes a utilizar para el generador de recetas de entrenamiento
        @max_ingredientes: Máximo de ingredientes a utilizar para el generador de recetas de entrenamiento
        @min_unidades: Al generar un dataset de entrenamiento ficticio, la cantidad mínima de unidades a utilizar por el algoritmo
        @max_unidades: Al generar un dataset de entrenamiento ficticio, la cantidad máxima de unidades a utilizar por el algoritmo
        @learning_rate: La tasa de aprendizaje utilizada por el optimizador (Adam)
        @batch_size: El tamaño de los lotes de entrenamiento
        @epochs: El número de épocas a entrenar el modelo
        @version: Un número interno que utilizamos para versionar nuestros modelos
        @kernels: Un factor base para el número de kernels o filtros de las capas de convolución que será pasado
                  al método GenerarModeloRegresionCNN (el cuál este método utiliza)
        @verbose: Si es True, imprime información del proceso de entrenamiento
        @save: Indica si se guardará automáticamente el modelo h5 en disco
        @savenumpy: Indica si al generar arreglos numpy de entrenamiento, se guardarán en disco como archivos npy
        
        Devuelve: Una tupla con modelo entrenado y el history del entrenamiento -> (model, history)
        """

        if df_training != '':
            npy_training = pd.read_csv(self.basedir + 'datasets/' + df_training, encoding = "ISO-8859-1").to_numpy()           
            print('Cargado dataset de entrenamiento:', self.basedir + 'datasets/' + df_training)
            recetas_train = []
            self.NUM_RECETAS = len(npy_training)
            print(self.NUM_RECETAS, 'recetas encontradas.')
            for i in range(len(npy_training)):
                row = npy_training[i]
                nombre = row[2]
                kcal = float(row[3])
                gramos_carb = float(row[4])
                gramos_proteina = float(row[5])
                gramos_grasa = float(row[6])


                recetas_train.append([nombre, round(kcal,2), 
                                    round(gramos_carb,2), 
                                    round(gramos_proteina,2), 
                                    round(gramos_grasa,2)]
                                    )

            recetas_train = np.array(recetas_train)             
            x, y = self.calcular_feature_vecs(recetas_train, max_len=self.EMB_SIZE, save=savenumpy, verbose=verbose)

        else:

            # Cargar los arrays de disco
            x, y = self.CargarNumpyRecetas(self.NUM_RECETAS, self.EMB_SIZE, verbose=verbose)

            if len(x)== 0 or len(y)==0:
                dataset_entrenamiento = self.generar_dataset_entrenamiento_nut(df_nutricionales=df_nutricionales,
                                                                    numero_recetas=self.NUM_RECETAS, 
                                                                    min_ingredientes=min_ingredientes, 
                                                                    max_ingredientes=max_ingredientes,
                                                                    min_unidades=min_unidades, max_unidades=max_unidades)

               
                x, y = self.calcular_feature_vecs(dataset_entrenamiento, max_len=self.EMB_SIZE, save=savenumpy, verbose=verbose)

        if (df_test == ''): #Si no proporcionas un dataframe de test, generarlo:
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
        else:
            x_train = x
            y_train = y 
            x_test, y_test = self.CargarNumpyRecetas(9, self.EMB_SIZE, verbose=verbose, sufix='_TEST')
            
            if len(x_test)==0 or len(y_test)==0:
                print('Procesando dataset de testing...')          
                array = self.procesar_dataset_validacion(df_test)
                x_test, y_test = self.calcular_feature_vecs(array, max_len=self.EMB_SIZE, save=True, verbose=verbose, sufix='_TEST')
            
        if (verbose): 
            if (df_val==''):
                x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8)
            else:      
                x_val, y_val = self.CargarNumpyRecetas(7, self.EMB_SIZE, verbose=verbose, sufix='_VAL') 
                if len(x_val)==0 or len(y_val)==0:  
                    print('Procesando dataset de validación...')         
                    array2 = self.procesar_dataset_validacion(df_val)
                    x_val, y_val = self.calcular_feature_vecs(array2, max_len=self.EMB_SIZE, save=True, verbose=verbose, sufix='_VAL')

        # Utilizamos la utilería Dataset de TensorFlow para que gestione la alimentación de los datasets de 
        # entrenamiento y no aparezcan errores por desbordamiento de memoria RAM o de RAM de GPU     
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        self.modeloCNN = self.GenerarModeloRegresionCNN(input_shape=(x_train.shape[1]), 
                                                        emb_size=self.EMB_SIZE, kernels=kernels,
                                                        numero_salidas=y_train.shape[1])

        # Utilizaremos como métrica de pérdida el MAE, ya que es un problema de regresión
        self.modeloCNN.compile(Adam(learning_rate=learning_rate), loss="mean_absolute_error", metrics=['mae'])
        
        if (verbose): self.modeloCNN.summary()

        # El modelo se guardará en la carpeta basedir + 'Modelos/'
        archivoC = self.basedir + 'Modelos/Modelo_Nut_FV_DistilBERT_0'+str(version)+'_EMBED-'+ str(self.EMB_SIZE) +'_CNN.h5'

        check_fileC = os.path.isfile(archivoC)

        if check_fileC:
            self.modeloCNN = tf.keras.models.load_model(archivoC)
       
        history = self.modeloCNN.fit(train_dataset,
                                batch_size = batch_size,
                                epochs = epochs,
                                initial_epoch=initial_epoch,                                
                                steps_per_epoch=steps_per_epoch,
                                validation_data=test_dataset,                               
                                verbose=verbose)

        if (save): 
            self.modeloCNN.save(archivoC)
            print('modelo guardado en:', archivoC)

        if (verbose): self.EvaluarModeloRegresion(self.INFO_COLS, history, x_val, y_val, self.modeloCNN)   

        return self.modeloCNN, history


    ############################################################################################################
    ############################################################################################################
    ## CARGAR UN ARCHIVO .H5 DE UN MODELO EN PARTICULAR 
    ############################################################################################################
    ############################################################################################################

    def CargarModelo(self, emb_size=128, version=4):
        """
        Carga un modelo existente desde archivo h5, utilizando el embedding size y la versión para
        armar el nombre del archivo y cargarlo en la variable modeloCNN de esta clase. 
        
        El formato es: 
            Modelo_Nut_FV_DistilBERT_[VERSION]_EMBED-[EMB_SIZE]_CNN.h5
            Donde [VERSION] es el número de versión que estamos entrenando o probando.
                  [EMB_SIZE] el tamaño de embeddings que se utilizó para codificar el texto con DistilBERT
        """

        archivoC = self.basedir + 'Modelos/Modelo_Nut_FV_DistilBERT_0'+str(version)+'_EMBED-'+ str(emb_size) +'_CNN.h5'
        check_fileC = os.path.isfile(archivoC)

        if check_fileC:
            self.modeloCNN = tf.keras.models.load_model(archivoC)
            self.EMB_SIZE = emb_size
            print('Modelo', archivoC, 'cargado con éxito.')
        else:
            print('No se encontró el modelo', archivoC)
            print('Puedes crear uno nuevo con el método EntrenarModelo()\n')
        return



    ############################################################################################################
    ############################################################################################################
    ## FUNCIÓN PARA PREDECIR VALORES NUTRICIONALES DADA UNA LISTA DE INGREDIENTES DE RECETAS
    ############################################################################################################
    ############################################################################################################
    def PredecirInfoNutricional(self, lista_ingredientes, 
                                INFO_COLS=None, modelo=None, 
                                emb_size=128, verbose=True):
        """
        Realiza una inferencia de los valores nutrimentales dada una lista de ingredientes.

        Parámetros:
        @lista_ingredientes: Un array con los ingredientes de cada receta
        @INFO_COLS: La lista de columnas o features que se están infiriendo por regresión.
        @modelo: El modelo de regresión que se utiliza para la inferencia (si es None, se utiliza self.modeloCNN)
        @emb_size: El espacio de embbedings que se utilizó para el encoding (es importante que sea el mismo que el modelo utilizado para las inferencias)
        
        Devuelve:
        @Una lista tipo diccionario con las predicciones de la información nutricional basada en la lista de ingredientes 
        de las recetas proporcionada.        
        """

        if (modelo == None): modelo = self.modeloCNN
        if (INFO_COLS== None): INFO_COLS = self.INFO_COLS

        # Tokenizar y sacar feature vector
        inputs = []
        result = []
        if (verbose):
            print('Extrayendo vectores de características de los ingredientes...\n')
            for i in tqdm(range(len(lista_ingredientes))):
                ingredientes = lista_ingredientes[i]
                inputs.append(np.reshape(self.get_feature_vectors(ingredientes, max_len=emb_size).flatten(), newshape=(1,-1)))
        else:            
            for i in range(len(lista_ingredientes)):
                ingredientes = lista_ingredientes[i]
                inputs.append(np.reshape(self.get_feature_vectors(ingredientes, max_len=emb_size).flatten(), newshape=(1,-1)))

        inputs = np.array(inputs)
        inputs = np.reshape(inputs, newshape=(len(lista_ingredientes), inputs.shape[2]))

        # Predecir con el modelo entrenado
        preds = modelo.predict(inputs)

        for i_pred in range(len(preds)):
            vals = preds[i_pred]
            nutricion = dict()
            nutricion['ingredientes'] = lista_ingredientes[i_pred]
            for j in range(len(INFO_COLS)):
                nutricion[INFO_COLS[j]] = vals[j]
            result.append(nutricion)    

        if (verbose):
            for i in range(len(result)):           
                row = result[i]
                for entrada in row:            
                    print(entrada,':', row[entrada])
                print('---------------------------------------------------------------------------\n')               

        return result



    ############################################################################################################
    ############################################################################################################
    ## FILTRADO DE RECETAS DE ACUERDO A SU CALIDAD NUTRICIONAL
    ############################################################################################################
    ############################################################################################################

    def Calcular_InfoNutricional_from_List(self, lista_ingredientes_recetas, verbose=True):
        """
        Calcula la información nutricional y los costos de acuerdo al una lista de strings
        con los ingredientes de recetas

        Parámetros:
        @lista_ingredientes_recetas: Una lista que contiene strings con listas de ingredientes
        @verbose: Indica si se imprimen mensajes del proceso
        
        Devuelve:
        Un dataframe con lo siguiente:
          kcal, proteinas_gr, carbohidratos_gr, grasas_gr, puntaje_platillo
        """

        # Por cada receta:
        # 1. Extraer ingredientes individuales
        # 2. Calcular sus valores nutricionales
        # 3. Agregarlos al dataframe resultante
        if (verbose): print('Calculando información nutricional y costos... \n')

        Calorias = []
        Proteinas = []
        Carbs = []
        Grasas = []
        Califs = []
        

        nutricion = self.PredecirInfoNutricional(lista_ingredientes_recetas, self.INFO_COLS, self.modeloCNN, 
                                                 self.EMB_SIZE, verbose=verbose)

        for i in range(len(nutricion)):
            row = nutricion[i]   

            kcal = round(float(row[self.INFO_COLS[0]]),2)            
            Calorias.append(kcal)

            carbs = round(float(row[self.INFO_COLS[1]]),2)
            Carbs.append(carbs)
            
            prots = round(float(row[self.INFO_COLS[2]]),2)
            Proteinas.append(prots)
            
            grasas = round(float(row[self.INFO_COLS[3]]),2)
            Grasas.append(grasas)

            # Calificar el platillo de acuerdo a:
            # RANGO_CARBOHIDRATOS
            # RANGO_PROTEINAS
            # RANGO_GRASAS

            # Calcular los porcentajes:
            # grasas = 9 kcal x gramo, carbs = 4 kcal * gramo, proteinas = 4 kcal * gramo

            p_carb = round(((carbs*4) / kcal)*100)
            p_prot = round(((prots*4) / kcal)*100)
            p_grasas = round(((grasas*4) / kcal)*100)
            
            calificacion_receta = 0.0
            if p_carb in self.RANGO_CARBOHIDRATOS: calificacion_receta += 1
            if p_prot in self.RANGO_PROTEINAS: calificacion_receta += 1
            if p_grasas in self.RANGO_GRASAS: calificacion_receta += 1

            calificacion_receta = round(calificacion_receta / 3, 1)     
            Califs.append(calificacion_receta)    

        dfFiltrados = pd.DataFrame(list(zip(lista_ingredientes_recetas, Calorias, Proteinas, Carbs, Grasas, Califs)),
                                   columns=['ingredientes', 'kcal', 'proteinas_gr', 'carbs_gr', 'grasas_gr', 'puntaje_platillo']) 
                            
        return dfFiltrados


    ############################################################################################################
    ############################################################################################################
    ## CÁLCULO DE LA INFORMACIÓN NUTRICIONAL DE UN DATASET UTILIZANDO EL MODELO DE REGRESIÓN:
    ############################################################################################################
    ############################################################################################################

    def Calcular_InfoNutricional(self, dfFiltrados=None, col_ingredientes='ingredientes', 
                                                        verbose=True, inline=False):
        """
        Calcula la información nutricional y los costos de acuerdo al dataset
        de información nutricional y al dataset de la canasta básica

        Parámetros:
        @dfFiltrados: Un dataframe ya filtrado de preferencia sobre del cuál se insertarán columnas con info. nutricional (Si es none se utiliza self.DF_RecetasFiltradas)
        @col_ingredientes: Nombre de la columna que contiene los ingredientes del dataframe de entrada
        @verbose: Indica si se imprimen mensajes del proceso
        @inline: Si es true, sobreescribe la variable DF_FILTRADOS de la clase
        
        Devuelve:
        Una copia del dataframe filtrado de entrada con nuevas columnas:
          kcal, proteinas_gr, carbohidratos_gr, grasas_gr, puntaje_platillo

          La variable puntaje_platillo es un factor porcentual entre 0 y 1 que indica que tan bien cumple
          con la regla de las proporciones saludables de carbohidratos, proteínas y grasas de gramos por cantidad de energía (kcal)
        """

        if (dfFiltrados == None): dfFiltrados = self.DF_RecetasFiltradas.copy()

        # Para poder llamar a este método, debe haberse ejecutado antes ProcesarRecetas
        if (len(dfFiltrados) <= 0):
            print('No se encontraron recetas pre-seleccionadas.\nEjecuta el método FiltrarRecetario_por_CanastaBasica()\n')
            return
        

        # Por cada receta:
        # 1. Extraer ingredientes individuales
        # 2. Calcular sus valores nutricionales
        # 3. Agregarlos al dataframe resultante
        if (verbose): print('Calculando información nutricional y costos... \n')

        Calorias = []
        Proteinas = []
        Carbs = []
        Grasas = []
        Califs = []

        recetas = [str(dfFiltrados.iloc[i][col_ingredientes]).strip() for i in range(len(dfFiltrados))]

        nutricion = self.PredecirInfoNutricional(recetas, self.INFO_COLS, self.modeloCNN, 
                                                 self.EMB_SIZE, verbose=verbose)

        for i in range(len(nutricion)):
            row = nutricion[i]   

            kcal = round(float(row[self.INFO_COLS[0]]),2)            
            Calorias.append(kcal)

            carbs = round(float(row[self.INFO_COLS[1]]),2)
            Carbs.append(carbs)
            
            prots = round(float(row[self.INFO_COLS[2]]),2)
            Proteinas.append(prots)
            
            grasas = round(float(row[self.INFO_COLS[3]]),2)
            Grasas.append(grasas)
            

            # Calificar el platillo de acuerdo a:
            # RANGO_CARBOHIDRATOS
            # RANGO_PROTEINAS
            # RANGO_GRASAS

            # Calcular los porcentajes:
            # grasas = 9 kcal x gramo, carbs = 4 kcal * gramo, proteinas = 4 kcal * gramo

            p_carb = round(((carbs*4) / kcal)*100)
            p_prot = round(((prots*4) / kcal)*100)
            p_grasas = round(((grasas*4) / kcal)*100)
            
            calificacion_receta = 0.0
            if p_carb in self.RANGO_CARBOHIDRATOS: calificacion_receta += 1
            if p_prot in self.RANGO_PROTEINAS: calificacion_receta += 1
            if p_grasas in self.RANGO_GRASAS: calificacion_receta += 1

            calificacion_receta = round(calificacion_receta / 3, 1)     
            Califs.append(calificacion_receta)     
                            
        dfFiltrados['kcal'] = Calorias
        dfFiltrados['proteinas_gr'] = Proteinas
        dfFiltrados['carbohidratos_gr'] = Carbs
        dfFiltrados['grasas_gr'] = Grasas
        dfFiltrados['puntaje_platillo'] = Califs


        if (inline): self.DF_RecetasFiltradas = dfFiltrados

        return dfFiltrados