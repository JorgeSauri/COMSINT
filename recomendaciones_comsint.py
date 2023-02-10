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

import random
from transformers import TFDistilBertModel, DistilBertTokenizerFast
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, Conv1D, MaxPool1D, Reshape, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import config
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ReduceLROnPlateau
import os.path


class Recomendador():


    NUM_RECETAS = 5000
    EMB_SIZE = 512
    VOCAB_SIZE = 768
    INFO_COLS = ['kcal','carbohydrate', 'protein', 'total_fat', 'sugars', 'fiber']


    def __init__(self,
                 fuente='recetas.csv',
                 nutricion='nutricion.csv',
                 canasta='canasta_basica.csv',
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
            @fuente: Ruta y archivo csv del recetario.
            @nutricion: Ruta y archivo csv del dataset de información nutricional
            @canasta: Ruta y archivo csv de la canasta básica
            @encoding: Tipo de codificación del archivo (utf-8 o iso-8859-1)
        """

        # Si hay hardware de GPU, inicializarlo y configurarlo para que administre bien la memoria del hardware
        physical_devices = config.list_physical_devices('GPU') 
        config.experimental.set_memory_growth(physical_devices[0], False)        
        tf.config.set_visible_devices([], 'GPU')

        # cargamos el modelo entrenado en español
        self.nlp = spacy.load("es_core_news_md")

        # Diccionario de medidas más comunes en recetas
        self.Medidas = {
            'miligramos': ['mg ', 'miligramo ', 'miligramos ', 'mgr ', 'mg.', 'mgr.'],
            'gramos': ['gramos ', 'gr ', 'g ', 'gram ', 'grams ', 'gr.', 'g.', 'gram.', 'grams.'],
            'onzas': ['onza ', 'onzas ', 'oz ', 'ozs ', 'onza.', 'onzas.', 'oz.', 'ozs.'],
            'kilos': ['kilo ', 'kilos ', 'kg ', 'k ', 'kgr ', 'kilo.', 'kilos.', 'kg.', 'k.', 'kgr.'],
            'mililitros': ['mililitro ', 'mililitros ', 'ml ', 'mltr ', 'mltrs ', 'ml.', 'mltr.', 'mltrs.'],
            'litros': ['litro ', 'litros ', 'l ', 'lt ', 'ltr ', 'ltrs ', 'l.', 'lt.', 'ltr.', 'ltrs.'],
            'piezas': ['pieza ', 'piezas ', 'unidad ', 'unidades ', 'pz ', 'pza ', 'pz.', 'pza.'],
            'tazas': ['taza ', 'tazas ', 'tza ', 'tz ', 'cup ', 'cups ', 'tza.', 'tz.'],
            'cucharadas': ['cucharada ', 'cucharadas ', 'cuch ', 'cda ', 'cdas ', 'cuch.', 'cda.', 'cdas.', 'tbsp ',
                           'tbsp.'],
            'cucharaditas': ['cucharadita ', 'cucharaditas ', 'cdta ', 'cdtas ', 'cdta.', 'cdtas.', 'tsp ', 'tsp.']
        }

        self.stopwords = ["el", "para", "con", "en", ",", "contra",
                          "de", "del", "la", "las", "los", "un",
                          "una", "unos", "unas", "o", "ó", "y"]

        # Dataframes:
        self.DF_RecetasFiltradas = None

        if (fuente != ''): self.df_recetario = pd.read_csv(fuente, encoding=encoding)
        if (nutricion != ''): self.df_nutricion = pd.read_csv(nutricion, encoding=encoding)
        if (canasta != ''): self.df_canasta = pd.read_csv(canasta, encoding=encoding)

        

    ##################################################################
    # Utilerías:
    ##################################################################
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

        for medida in self.Medidas:
            for abr in self.Medidas[medida]:
                index = cadena.find(abr)
                if index > -1:
                    result = medida
                    break

        return result

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
                    unidad = self.encontrar_unidades(cad.split(token.text)[1])
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
                            unidad = self.encontrar_unidades(cad.split(token.text)[1])
                            unidades.append(unidad)
                            ingrediente_texto = cad.split(token.text)[1]
                            ingrediente_texto = self.LimpiarString(ingrediente_texto)
                            ingredientes_texto.append(ingrediente_texto)
                            break

                            # Devuelve las listas
        return cantidades, unidades, ingredientes_texto

    ##################################################################
    # Métodos especiales para el filtrado de recetas por canasta básica:
    ##################################################################
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
                Sim.append(similaridad)

        dfFiltrados = pd.DataFrame(list(zip(Platillos, Ingredientes, Sim)),
                                   columns=['nombre_del_platillo', 'ingredientes', 'similitud'])

        dfFiltrados = dfFiltrados.sort_values(by=['similitud'], ascending=False)[:max_rows]

        print(' \n\n', len(dfFiltrados), 'platillos encontrados con similitud mayor a', similitud)

        # Guardamos el dataframe en una variable de la clase, y también la regresamos
        self.DF_RecetasFiltradas = dfFiltrados

        if (verbose): return dfFiltrados


    ##################################################################
    # Métodos especiales para el modelo de cálculo de info nutrimental:
    ##################################################################

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

    def get_feature_vectors(self, ingredient_list, max_len=64):
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

    def generar_dataset_entrenamiento(self, 
                                    df_nutricionales='nutricion.csv', 
                                    encoding='ISO-8859-1',
                                    usecols=['nombre', 'kcal','carbohydrate', 'protein', 'total_fat', 'sugars', 'fiber'],
                                    min_ingredientes = 1,
                                    max_ingredientes = 5, 
                                    numero_recetas=100):
        """
        Regresa un NumPy Array para entrenar un modelo de regresión.
        Por defecto se toman las columnas: 'nombre', 'kcal','carbohydrate', 'protein', 'total_fat', 'sugars', 'fiber'
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
            dataset = generar_dataset_entrenamiento(numero_recetas=1000, min_ingredientes=5, max_ingredientes=10)
        """

        df = pd.read_csv(df_nutricionales, encoding=encoding, usecols=usecols)
        
        print('Generando', numero_recetas,' recetas aleatorias...\n')
        
        RecetaRandom = []

        for i_recetas in tqdm(range(numero_recetas)):
                nombre = ''
                kcal = 0.0
                gramos_carb = 0.0
                gramos_proteina = 0.0
                gramos_grasa = 0.0
                gramos_azucar = 0.0
                gramos_fibra = 0.0

                for i_ingredientes in range(np.random.randint(min_ingredientes, max_ingredientes+1)):
                    # Elegir un ingrediente al azar el dataframe de nutricionales
                    i_rand = np.random.randint(len(df))
                    cant_rand = round(np.random.ranf() * np.random.randint(1,10), 2)
                    row_alimento = df.iloc[i_rand]
                    nombre += str(cant_rand) + 'gr de ' + row_alimento['nombre'].replace(',', ' ').strip() + ', '
                    kcal += cant_rand * float(row_alimento['kcal'])       
                    gramos_carb += cant_rand * float(row_alimento['carbohydrate'].replace(' ', '').split('g')[0])
                    gramos_proteina += cant_rand * float(row_alimento['protein'].replace(' ', '').split('g')[0])                               
                    gramos_grasa += cant_rand * float(row_alimento['total_fat'].replace(' ', '').split('g')[0])
                    gramos_azucar += cant_rand * float(row_alimento['sugars'].replace(' ', '').split('g')[0])             
                    gramos_fibra += cant_rand * float(row_alimento['fiber'].replace(' ', '').split('g')[0])            
                        
                nombre = nombre[:-2]
                RecetaRandom.append([nombre, round(kcal,2), round(gramos_carb,2), round(gramos_proteina,2), 
                                    round(gramos_grasa,2), round(gramos_azucar,2), round(gramos_fibra,2)])
                
                
        result = np.array(RecetaRandom)


        return result

    def calcular_feature_vecs(self, array_recetas, max_len=128, save=True, verbose=True):

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
            np.save('datasets/numpy/' + str(len(array_recetas)) + '_recetas_random_EMBED-'+ str(max_len) +'_DATA_X', result_x)
            np.save('datasets/numpy/' + str(len(array_recetas)) + '_recetas_random_EMBED-'+ str(max_len) +'_DATA_Y', result_y)

        return result_x, result_y

    def GenerarModeloRegresionCNN(self, input_shape, emb_size, numero_salidas):
            """
            Devuelve un modelo de CNN 1D para aprender 
            los patrones de ingredientes y sus valores nutricionales.

            Parámetros:
            @emb_size: El tamaño de embbeding que se utilizó (el vocab es 768)
            @numero_salidas: El número de columnas o valores que aprenderá a predecir.

            Devuelve: 
            Una instancia de la clase tensorflow.keras.Model

            Ejemplo:
            modelo = GenerarModelo(emb_size=512, numero_salidas=y_train.shape[1])
            modelo.compile(RMSprop(learning_rate=1e-5), loss="mean_absolute_error", metrics=['accuracy'])
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

            # Capas de convolución
            cnn = Conv1D(64, 5, activation='relu', name='CONV_1')(reshaped)       
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_1')(cnn)
            cnn = Conv1D(64, 3, activation='relu', name='CONV_2')(cnn)
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_2')(cnn)
            cnn = Conv1D(64, 3, activation='relu', name='CONV_3')(cnn)
            cnn = MaxPool1D(pool_size=2, strides=1, padding='valid', name='POOLING_3')(cnn)
            cnn = Dropout(0.2)(cnn)
            cnn = Flatten()(cnn)

            # Capas densamente conectadas para aprender características y patrones 

            x = Dense(256, activation='relu')(cnn)        
            x = Dense(128, activation='relu')(x)  
            x = Dense(64, activation='relu')(x)   
            x = Dropout(0.2)(x)     
            x = Flatten()(x)
            output_tensor = Dense(numero_salidas, name='CapaSalida')(x)

            model = Model(inputs=input_tensor, outputs=output_tensor, name="ModeloCNNNut")
            model.build(input_shape)

            self.modeloCNN = model

            return model

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

        pd.DataFrame(history.history).plot()
        plt.show()

        scores = modelo.evaluate(x_val, y_val)
        print(scores)

        test_predictions = modelo.predict(x_val)
        sum_error = []
        for i in range(len(y_val)):
                for j in range(len(INFO_COLS)):
                    error = y_val[i][j] / test_predictions[i][j]
                    if (error > 1): error = error - 2
                    sum_error.append(error)
                    print('receta',i, INFO_COLS[j]+'_true:', y_val[i][j], INFO_COLS[j]+'_pred:', 
                            test_predictions[i][j], ' precisión:', round(error * 100, 1),'%')
                print('---------------------------------------------------------------------------')
        print('Precisión promedio aprox. = ', round(np.mean(sum_error)*100,2),'%')
        return

    def CargarNumpyRecetas(self, basedir, NUM_RECETAS, EMB_SIZE, verbose=True):
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

        archivoX = basedir.strip() + str(NUM_RECETAS) + '_recetas_random_EMBED-'+ str(EMB_SIZE) +'_DATA_X.npy'
        archivoY = basedir.strip() + str(NUM_RECETAS) + '_recetas_random_EMBED-'+ str(EMB_SIZE) +'_DATA_Y.npy'
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

    def EntrenarModelo(self, df_nutricionales='datasets/nutricion.csv', 
                       min_ingredientes=5, max_ingredientes=10,
                       batch_size = 8,
                       epochs = 20,
                       verbose=True, save=True):
        """
        Entrenar el modelo de cálculo de información nutricional

        Parámetros:
        @df_nutricionales: El dataset de valores nutricionales con el que se arma el dataset de entrenamiento
        @min_ingredientes: Mínimo de ingredientes a utilizar para el generador de recetas de entrenamiento
        @max_ingredientes: Máximo de ingredientes a utilizar para el generador de recetas de entrenamiento
        @batch_size: El tamaño de los lotes de entrenamiento
        @epochs: El número de épocas a entrenar el modelo
        @verbose: Si es True, imprime información del proceso de entrenamiento
        @save: Indica si se guardará automáticamente el modelo h5 en disco

        Devuelve: El modelo entrenado y el history del entrenamiento    
        """

        # Cargar los arrays de disco
        x, y = self.CargarNumpyRecetas('datasets/numpy/', self.NUM_RECETAS, self.EMB_SIZE, verbose=verbose)

        if len(x)== 0 or len(y)==0:
            dataset_entrenamiento = self.generar_dataset_entrenamiento(df_nutricionales=df_nutricionales,
                                                                numero_recetas=self.NUM_RECETAS, 
                                                                min_ingredientes=min_ingredientes, 
                                                                max_ingredientes=max_ingredientes)

            dataset_entrenamiento[np.random.randint(len(dataset_entrenamiento))]      

            x, y = self.calcular_feature_vecs(dataset_entrenamiento, max_len=self.EMB_SIZE, save=False, verbose=verbose)

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
        if (verbose): x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        self.modeloCNN = self.GenerarModeloRegresionCNN(input_shape=(x_train.shape[1]), 
                                                        emb_size=self.EMB_SIZE, 
                                                        numero_salidas=y_train.shape[1])

        self.modeloCNN.compile(Adam(learning_rate=1e-4), loss="mean_absolute_error", metrics=['mae'])
        #if (verbose): self.modeloCNN.summary()

        archivoC = 'Modelos/Modelo_Nut_FV_DistilBERT_02_EMBED-'+ str(self.EMB_SIZE) +'_CNN.h5'

        check_fileC = os.path.isfile(archivoC)

        if check_fileC:
            self.modeloCNN = tf.keras.models.load_model(archivoC)
            
        history = self.modeloCNN.fit(train_dataset,
                                batch_size = batch_size,
                                epochs = epochs,
                                validation_data=test_dataset,
                                verbose=verbose)

        if (save): 
            self.modeloCNN.save(archivoC)
            print('modelo guardado en:', archivoC)

        if (verbose): self.EvaluarModeloRegresion(self.INFO_COLS, history, x_val, y_val, self.modeloCNN)   

        return self.modeloCNN, history

    def CargarModelo(self, basedir='', emb_size=512):

        archivoC = basedir + 'Modelo_Nut_FV_DistilBERT_02_EMBED-'+ str(emb_size) +'_CNN.h5'
        check_fileC = os.path.isfile(archivoC)

        if check_fileC:
            self.modeloCNN = tf.keras.models.load_model(archivoC)
            self.EMB_SIZE = emb_size
            print('Modelo', archivoC, 'cargado con éxito.')
        else:
            print('No se encontró el modelo', archivoC)
            print('Puedes crear uno nuevo con el método EntrenarModelo()\n')
        return

    def PredecirInfoNutricional(self, lista_ingredientes, INFO_COLS=None, modelo=None, emb_size=512, verbose=True):
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

    ##################################################################
    # Filtrado de recetas por mejor calidad nutrimental:
    ##################################################################
    def Calcular_InfoNutricional(self, dfFiltrados=None, col_ingredientes='ingredientes', verbose=True, inline=False):
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
          kcal, proteinas_gr, carbohidratos_gr, grasas_gr, fibra_gr, azucar_gr, costo_total_min, costo_total_max
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
        Fibras = []
        Azucares = []

        recetas = [str(dfFiltrados.iloc[i][col_ingredientes]).strip() for i in range(len(dfFiltrados))]

        nutricion = self.PredecirInfoNutricional(recetas, self.INFO_COLS, self.modeloCNN, 
                                                 self.EMB_SIZE, verbose=verbose)

        for i in range(len(nutricion)):
            row = nutricion[i]               
            Calorias.append(round(float(row[self.INFO_COLS[0]]),2))
            Carbs.append(round(float(row[self.INFO_COLS[1]]),2))
            Proteinas.append(round(float(row[self.INFO_COLS[2]]),2))
            Grasas.append(round(float(row[self.INFO_COLS[3]]),2))
            Azucares.append(round(float(row[self.INFO_COLS[4]]),2))
            Fibras.append(round(float(row[self.INFO_COLS[5]]),2))
                            
        dfFiltrados['kcal'] = Calorias
        dfFiltrados['proteinas_gr'] = Proteinas
        dfFiltrados['carbohidratos_gr'] = Carbs
        dfFiltrados['grasas_gr'] = Grasas
        dfFiltrados['fibra_gr'] = Fibras
        dfFiltrados['azucar_gr'] = Azucares

        if (inline): self.DF_RecetasFiltradas = dfFiltrados

        return dfFiltrados