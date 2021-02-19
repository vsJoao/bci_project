"""
# %% Reorganização do Dataset

# Esse código tem por objetivo organizar o dataset utilizado com o objetivo de
# facilitar a escrita dos códigos futuros, deixando-os mais limpos e com uma
# leitura mais fácil. Além de estabelecer um padrão para que os dados entrem no
# algoritmo. O dataset é intitulado _Four class motor imagery (001-2014)
# e pode ser baixado pelo link: (http://bnci-horizon-2020.eu/database/data-sets).

# No site os arquivos encontram-se em formato .mat para ser aberto na linguagem
# MatLab, logo é preciso fazer um tratamento nesses dados antes de utilizá-los
# em pyhton. Ao carregar o arquivo com o comando '''readmat''' é carregada uma
# variável dicionário na qual o indice de interesse é o 'data'. Após carregar
# essa instância do arquivo, é criada uma variável do tipo array de 8 dimensões.

# %% Carregamento das bibliotecas e nome dos arquivos do dataset:
"""
from configs.database_names import *

from scipy.io import loadmat
import numpy as np
import os
import mne

"""
Descrição do Dataset

O resultado do carregamento feito por a seguir será um array (data) de 7 dimensões
que pode ser acessado pelos" seguintes indices:


"file = loadmat("A01T.mat")
data = file['data']

**data[id1][id2][id3][id4][id5][id6][id7]**

Cada um dos indices desse array são referentes a uma informação do dataset:
* id1: Indexa todo o conjunto de informações deve ser deixado como zero
* id2: Indexa o número de um dos 9 conjuntos de testes que foram realizados (0-8)
* id3: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
* id4: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
* id5: Esse indice indica um tipo de informação específico que será indexado pelas próxi- 
       mas dimensões do array:
    * 0: Dados de amostragem (X)
    * 1: Guarda o indice de onde inciam os testes nos dados de amostragem (trial)
    * 2: Classe referente a cada um dos testes indexados anteriormente (y)
    * 3: Frequencia de amostragem do dataset (fs)
    * 4: Nomes de cada uma das classes indexadas enteriormente (classes)
    * 5: Indica em quais testes indexados anteriormente estão presentes artefatos (Artifacts)
    * 6: Genero da pessoa que está sendo analisada (gender)
    * 7: Idade da pessoa que está sendo analisada (age)
* id6: Esse indice faz referencia ao indice da entrada anterior quando esta for um vetor
       (Para o caso de dados de amostragem, esse indice fará refencia a cada uma das
       amostragens feitas no experimento de acordo com a frequencia de amostragem estabe- 
       lecida em fs)
* id7: Esse indice está disponível apenas para os dados de amostragem e irá referenciar
       cada um dos 25 eletrodos utilizados na amostragem
"""

