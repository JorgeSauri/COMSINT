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
        print('Inicializando Recomendador de recetas para entrenamiento de precios...\n')
        self.Agente = Recomendador(basedir = self.basedir, fuente=recetario)
        self.Agente.CargarModelo(emb_size=emb_size, version=4)

        self.Agente.NUM_RECETAS = num_recetas
        self.Agente.EMB_SIZE = emb_size
        self.BATCHSIZE = batch_size
        self.ITER = it
        self.LR = lr
        if verbose==0:
            self.verbose = False
        if verbose==1:
            self.verbose = True
        if epochs == None:
            self.EPOCHS = self.Agente.NUM_RECETAS // self.BATCHSIZE
        else:
            self.EPOCHS = int(epochs)
        self.version = 4
        
        return
    

    def do_training(self):

        INITIAL_EPOCH = 0
        MINU = 3 
        MAXU = 11  

        Histories = []
        for iteracion in range(self.ITER):
            print('\nITERACIÓN:', iteracion+1)
            print('Entrenando desde epoch', INITIAL_EPOCH)
            print('------------------------------------------\n')
            modelo, history = self.Agente.EntrenarModeloPrecios(
                                            df_precios='lista_precios_profeco_2022.csv',                                   
                                            learning_rate=self.LR,
                                            version=self.version, 
                                            initial_epoch = INITIAL_EPOCH,                             
                                            epochs=INITIAL_EPOCH + self.EPOCHS,
                                            min_ingredientes=5, max_ingredientes=11,                                  
                                            min_unidades=MINU, max_unidades=MAXU,
                                            batch_size=self.BATCHSIZE,
                                            kernels=128,                                                                         
                                            save=True, verbose=self.verbose)
            INITIAL_EPOCH = history.epoch[-1]

            Histories.append(history)


        self. modelo = modelo
        self.Histories = Histories

        return 
    
    
    
    def Evaluar(self):

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
parser.add_argument('-verbose', dest='verbose', required=False,
                    help='Indica si se muestran mensajes de evaluación del entrenamiento 1=Si | 0=No (default=Si)')
parser.add_argument('-num_recetas', dest='num_recetas', required=False,
                    help='Número de recetas de entrenamiento (default=1000)')
parser.add_argument('-batch_size', dest='batch_size', required=False,
                    help='Batch size (default=32)')
parser.add_argument('-epochs', dest='epochs', required=False,
                    help='Epocas de entrenamiento')
parser.add_argument('-it', dest='it', required=False,
                    help='Iteraciones (default=1)')
parser.add_argument('-lr', dest='lr', required=False,
                    help='Learning rate (default=1e-4)')


# parsea los argumentos de la línea de comandos
args = parser.parse_args()

recetario = args.recetario
emb_size = args.emb_size
num_recetas = args.num_recetas
batch_size = args.batch_size
it = args.it
lr = args.lr
epochs = args.epochs
verbose = args.verbose


if verbose==None: 
	verbose = 1
else:
	if verbose>=0:
		verbose = int(verbose)
		if verbose>1: verbose = 1



if recetario == None: recetario = 'recetario_mexicano_small.csv'
if emb_size == None: emb_size = 128
if num_recetas == None: num_recetas = 1000
if batch_size == None: batch_size = 32
if it == None: it = 1
if lr == None: lr = 1e-3


emb_size = int(emb_size)
num_recetas = int(num_recetas)
batch_size = int(batch_size)
it = int(it)
lr = float(lr)


print('Recetario: ', recetario)
print('Numero de recetas:', num_recetas)
print('Batch size: ', batch_size)
print('Epocas:', epochs)
print('Learning rate:', lr)


## Entrenar:
trainer = Trainer()
trainer.do_training()
exit()
