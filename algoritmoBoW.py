from geral import *

import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from unidecode import unidecode
from joblib import dump, load

caminhoArquivos = 'arquivos/'

nomeArquivo = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text'
nomeColTexto2 = 'text_norm'
nomeColRotulo = 'toxic'

caminhoBoW = caminhoArquivos + 'bow/'  # Novo caminho para salvar o modelo

nomeArquivoVectorizer = caminhoArquivos + 'vectorizer.joblib'
nomeArquivoClassificador = caminhoArquivos + 'classificador.joblib'

if not os.path.exists(caminhoBoW):
    os.makedirs(caminhoBoW)

def lerArquivo():
    try:
        df = pd.read_csv(nomeArquivo)
        df = tratarDataframe(df)
        return df
    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "lerArquivo": \n' + str(e))

def tratarDataframe(df):
    try:
        print('Tamanho inicial do dataframe = ' + str(df.shape))

        # Unificando as colunas de texto e duplicando o rótulo para não perder a consistência de dados
        df_ColTexto1 = df[[nomeColTexto,  nomeColRotulo]]
        df_ColTexto2 = df[[nomeColTexto2, nomeColRotulo]]
        df_ColTexto2 = df_ColTexto2.rename(columns={nomeColTexto2: nomeColTexto})
        df_novo = pd.concat([df_ColTexto1, df_ColTexto2], ignore_index=True, axis=0, keys=None, names=[nomeColTexto, nomeColRotulo])

        # Removendo linhas com valores nulos
        print('Valores nulos = ' + str(df_novo.isnull().sum().sum()))
        df_novo = df_novo.dropna().reset_index(drop=True)

        # Substituindo caracteres acentuados por seus respectivos sem acentuação
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: unidecode(x))

        # Removendo caracteres que não sejam letras ou números
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

        print('Tamanho final do dataframe = ' + str(df_novo.shape))
        return df_novo
    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "tratarDataframe": \n' + str(e))

def treinarBoW(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador RandomForest
        print('\nInício do treinamento RandomForestClassifier com BoW')
        classificador = RandomForestClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento RandomForestClassifier com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarBoW": \n' + str(e))
    
def treinarPerceptron(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador Perceptron
        print('\nInício do treinamento Perceptron com BoW')
        classificador = Perceptron()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento Perceptron com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarPerceptron": \n' + str(e))
    
def treinarMLP(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador MLP
        print('\nInício do treinamento MLP com BoW')
        classificador = MLPClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento MLP com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarMLP": \n' + str(e))
    
def treinarDecisionTree(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador DecisionTree
        print('\nInício do treinamento DecisionTree com BoW')
        classificador = DecisionTreeClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento DecisionTree com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarDecisionTree": \n' + str(e))

def treinarAdaBoost(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador AdaBoost
        print('\nInício do treinamento AdaBoost com BoW')
        classificador = AdaBoostClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento AdaBoost com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarAdaBoost": \n' + str(e))
    
def treinarMultinomialNB(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador Multinomial Naive Bayes
        print('\nInício do treinamento Multinomial Naive Bayes com BoW')
        classificador = MultinomialNB()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento Multinomial Naive Bayes com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        
        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarMultinomialNB": \n' + str(e))
    
def treinarSGD(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador SGD
        print('\nInício do treinamento SGD com BoW')
        classificador = SGDClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento SGD com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

         # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)


    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarSGD": \n' + str(e))
    
def treinarPassiveAggressive(tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        # Treinando o classificador Passive Aggressive
        print('\nInício do treinamento Passive Aggressive com BoW')
        classificador = PassiveAggressiveClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento Passive Aggressive com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        # Calculando métricas para os resultados do tcc
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb)
        print('Taxa de Falsos Negativos (TFD):', tfd)
        print('Taxa de Falsos Positivos (TPF):', tpf)
        print('F-medida:', f_medida)

    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "treinarPassiveAggressive": \n' + str(e))

def validarTextoBoW(texto):
    try:
        # Carregando o vetorizador e o classificador
        vectorizer = load(nomeArquivoVectorizer)
        classificador = load(nomeArquivoClassificador)

        # Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)

        # Transformando o texto usando o vetorizador
        vetor_texto = vectorizer.transform([texto_processado])

        # Realizando a predição com o classificador
        predicao = classificador.predict(vetor_texto)

        return predicao <= 0.5
    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "validarTexto": \n' + str(e))

# Testar com um exemplo
texto_exemplo = "Este é um exemplo de texto para classificação."
resultado = validarTextoBoW(texto_exemplo)
print(f'O texto "{texto_exemplo}" é considerado não tóxico: {resultado}')
