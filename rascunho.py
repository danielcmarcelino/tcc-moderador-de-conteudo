from bibliotecas import *
import geral as g
import algoritmoBoW as bow
import algoritmoWord2Vec as w2v
import algoritmoTF as tfi

g.limparTela()

print('\n\n ----------------- INICIO DOS TREINAMENTOS -----------------')

print('\n\n ----------------- INICIO DO WORD2VECTOR ----------------- \n\n')
w2v.treinarModelos()
print('\n\n ----------------- FIM DO WORD2VECTOR ----------------- \n\n')

print('\n\n ----------------- INICIO DO BAG OF WORDS ----------------- \n\n')
bow.treinarModelos()
print('\n\n ----------------- FIM DO BAG OF WORDS ----------------- \n\n')

print('\n\n ----------------- INICIO DO TF-IDF ----------------- \n\n')
tfi.treinarModelos()
print('\n\n ----------------- FIM DO TF-IDF ----------------- \n\n')


print('\n\n ----------------- FIM DOS TREINAMENTOS -----------------')