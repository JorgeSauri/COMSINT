import argparse
import stablediffusion as sd
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# crea un objeto ArgumentParser
parser = argparse.ArgumentParser(description='Script para stablediffusion.')

# agrega un argumento para el archivo de recetas
parser.add_argument('-prompt', dest='prompt', required=True,
                    help='Prompt para generar una imagen')

# parsea los argumentos de la línea de comandos
args = parser.parse_args()


prompt = args.prompt

if __name__ == '__main__':
    print('Prompt:', prompt)
    # Crea un pipeline de SD.
    generador = sd.GeneradorImagenes()
    img = generador.GenerarImagen(prompt=prompt, negative_prompt='', 
                                  guidance_scale=7, num_inference_steps=25)
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.show()
    

