from geral import *
from algoritmoWord2Vec import treinarWord2Vec
from algoritmoBoW import treinarModelos  
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import Perceptron  # Importe a classe Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

limparTela()

# Lista de classificadores a serem treinados
classificadores = [
    AdaBoostClassifier(),
    RandomForestClassifier(),
    Perceptron(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    MultinomialNB(),
    SGDClassifier(),
    PassiveAggressiveClassifier()
]

for classificador in ['RFC', 'SVM', 'NB', 'ADA', 'PER', 'SGD', 'PA']:
    # Utilizando Word2Vec com Skip-Gram
    treinarWord2Vec(algoritmoTreinoWord2Vec=1, tamanhoTeste=0.3, algoritmoClassificador=classificador)
    print('\n\n')

    # Utilizando Word2Vec com CBOW
    treinarWord2Vec(algoritmoTreinoWord2Vec=0, tamanhoTeste=0.3, algoritmoClassificador=classificador)
    print('\n\n')

# Chama a função para treinar todos os modelos
treinarModelos(classificadores)
