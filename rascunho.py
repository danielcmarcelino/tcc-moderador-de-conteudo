from bibliotecas import *
import geral as g
import algoritmoBoW as bow
import algoritmoWord2Vec as w2v
import algoritmoTF as tfi

g.limparTela()

print('----------------- INICIO DOS TREINAMENTOS -----------------')

print('\n----------------- INICIO DO WORD2VECTOR -----------------')
w2v.treinarModelos()
print('----------------- FIM DO WORD2VECTOR -----------------\n')

print('\n----------------- INICIO DO BAG OF WORDS -----------------')
bow.treinarModelos()
print('----------------- FIM DO BAG OF WORDS -----------------\n')

print('\n----------------- INICIO DO TF-IDF -----------------')
tfi.treinarModelos()
print('----------------- FIM DO TF-IDF -----------------\n')

print('\n\n----------------- FIM DOS TREINAMENTOS -----------------')