from geral import *
from algoritmoWord2Vec import treinarWord2Vec
from algoritmoBoW import treinarBoW  


limparTela()

# Utilizando Word2Vec com Skip-Gram
treinarWord2Vec(algoritmoTreinoWord2Vec=1, tamanhoTeste=0.3)
print('\n\n')

# Utilizando Word2Vec com CBOW
treinarWord2Vec(algoritmoTreinoWord2Vec=0, tamanhoTeste=0.3)
print('\n\n')

# Utilizando Bag of Words
treinarBoW(tamanhoTeste=0.3)  

