from recomendaciones_comsint import Recomendador
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class Trainer:    
    basedir = ''

    def __init__(self):
        print('-------------------------------------------------------------')
        print('Inicializando Recomendador de recetas para entrenamiento...\n')
        self.Agente = Recomendador(basedir = self.basedir, fuente=recetario)
        self.Agente.CargarModelo(emb_size=emb_size, version=4)

        self.Agente.NUM_RECETAS = num_recetas
        self.Agente.EMB_SIZE = emb_size
        self.BATCHSIZE = batch_size
        self.ITER = it
        self.LR = lr
        self.rango_kcal = rango_kcal
        self.df_training = df_training
        self.df_test = df_test 
        self.df_val = df_val 

        self.version = 4
        
        return
    

    def do_training(self):

        INITIAL_EPOCH = 0
        self.EPOCHS = self.Agente.NUM_RECETAS // self.BATCHSIZE

        Histories = []
        for iteracion in range(self.ITER):
            MINU = 3 
            MAXU = 11  
            #INITIAL_EPOCH = 0
            
            MINK, MAXK = self.rango_kcal
            print('\nITERACIÓN:', iteracion+1)
            print('min unidades:',MINU, ' max unidades:', MAXU)
            print('min kcal:', MINK, ' max kcal:', MAXK)
            print('Entrenando desde epoch', INITIAL_EPOCH)
            print('------------------------------------------\n')
            modelo, history = self.Agente.EntrenarModelo(df_nutricionales='nutricion_mejorado.csv',
                                        df_training = self.df_training,
                                        df_test = self.df_test, 
                                        df_val = self.df_val,
                                        learning_rate=self.LR,
                                        version=self.version, 
                                        initial_epoch = INITIAL_EPOCH,                             
                                        epochs=INITIAL_EPOCH + self.EPOCHS, 
                                        batch_size=self.BATCHSIZE,
                                        kernels=128,                                             
                                        min_ingredientes=5, max_ingredientes=11,                                        
                                        min_unidades=MINU, max_unidades=MAXU,  
                                        min_kcal=MINK, max_kcal= MAXK,                             
                                        save=True, verbose=True)
            INITIAL_EPOCH = history.epoch[-1]

            Histories.append(history)


        self. modelo = modelo
        self.Histories = Histories

        return 
    
    
    
    def Evaluar(self):
        
        tf.keras.utils.plot_model(self.modelo, show_shapes=True)

        for i in range(len(self.Histories)):
            history = self.Histories[i]    
            pd.DataFrame(history.history).plot()
            plt.title('ITERACIÓN ' + str(i))
            plt.show()
            print('LOSS:',history.history['loss'][-1], ' -- MAE:', history.history['mae'][-1], 
                ' -- VAL_LOSS:', history.history['val_loss'][-1], ' -- VAL_MAE:', history.history['val_mae'][-1])
            print('----------------------------------------------------------')

        return
            




# crea un objeto ArgumentParser
parser = argparse.ArgumentParser(description='Script de entrenamiento para sistema de recomendaciones.')

# agrega un argumento para el archivo de recetas
parser.add_argument('-recetario', dest='recetario', required=False,
                    help='nombre del archivo CSV de datos con recetas')

# agregar un argumento para el tamaño de los embeddings
parser.add_argument('-emb_size', dest='emb_size', required=False,
                    help='Tamaño de embeddings del modelo (default=128)')

# parámetros para entrenamiento del modelo
parser.add_argument('-num_recetas', dest='num_recetas', required=False,
                    help='Número de recetas de entrenamiento (default=1000)')
parser.add_argument('-batch_size', dest='batch_size', required=False,
                    help='Batch size (default=32)')
parser.add_argument('-it', dest='it', required=False,
                    help='Iteraciones (default=1)')
parser.add_argument('-lr', dest='lr', required=False,
                    help='Learning rate (default=1e-4)')
parser.add_argument('-rango_kcal', dest='rango_kcal', required=False,
                    help='Rango kcal de recetas entrenamiento (tupla: (250, 2500) )')
parser.add_argument('-df_training', dest='df_training', required=False,
                    help='Dataframe de entrenamiento csv')
parser.add_argument('-df_test', dest='df_test', required=False,
                    help='Dataframe de testing csv')
parser.add_argument('-df_val', dest='df_val', required=False,
                    help='Dataframe de validación csv')


# parsea los argumentos de la línea de comandos
args = parser.parse_args()

recetario = args.recetario
emb_size = args.emb_size
num_recetas = args.num_recetas
batch_size = args.batch_size
it = args.it
lr = args.lr
rango_kcal = args.rango_kcal
df_training = args.df_training
df_test = args.df_test
df_val = args.df_val


if recetario == None: recetario = 'recetario_mexicano_small.csv'
if emb_size == None: emb_size = 128
if num_recetas == None: num_recetas = 1000
if batch_size == None: batch_size = 32
if it == None: it = 1
if lr == None: lr = 1e-4
if rango_kcal == None: 
    rango_kcal = (250, 2500)
else:
    rango_kcal = int(rango_kcal.split(',')[0][1:]), int(rango_kcal.split(',')[1][0:-1])

emb_size = int(emb_size)
num_recetas = int(num_recetas)
batch_size = int(batch_size)
it = int(it)
lr = float(lr)


if df_training==None: df_training=''
if df_test==None: df_test='recetas_test.csv'
if df_val==None: df_val='recetas_val.csv'


print('Recetario: ', recetario)
print('Numero de recetas:', num_recetas)
print('Batch size: ', batch_size)
print('Learning rate:', lr)
print('Rango de kcal:', rango_kcal)
print('Dataframe de entrenamiento:', df_training)
print('Dataframe de testing:', df_test)
print('Dataframe de validación:', df_val)

## Entrenar:
trainer = Trainer()
trainer.do_training()
exit()
