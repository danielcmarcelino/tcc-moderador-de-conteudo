from geral import *
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

# Chama a função para treinar todos os modelos
treinarModelos(classificadores)
