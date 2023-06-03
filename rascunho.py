from geral import *
from algoritmoWord2Vec import *

limparTela()
#Utilizando Skip-Gram
treinarWord2Vec(algoritmoTreinoWord2Vec=1, tamanhoTeste=0.3)
print('\n\n')
#Utilizando CBOW
treinarWord2Vec(algoritmoTreinoWord2Vec=0, tamanhoTeste=0.3)