from geral import *
import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from unidecode import unidecode
from joblib import dump, load

caminhoArquivos = 'arquivos/'
nomeArquivo = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text'
nomeColTexto2 = 'text_norm'
nomeColRotulo = 'toxic'
caminhoBoW = caminhoArquivos + 'bow/'
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
        df_ColTexto1 = df[[nomeColTexto,  nomeColRotulo]]
        df_ColTexto2 = df[[nomeColTexto2, nomeColRotulo]]
        df_ColTexto2 = df_ColTexto2.rename(columns={nomeColTexto2: nomeColTexto})
        df_novo = pd.concat([df_ColTexto1, df_ColTexto2], ignore_index=True, axis=0, keys=None, names=[nomeColTexto, nomeColRotulo])

        print('Valores nulos = ' + str(df_novo.isnull().sum().sum()))
        df_novo = df_novo.dropna().reset_index(drop=True)
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: unidecode(x))
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        print('Tamanho final do dataframe = ' + str(df_novo.shape))
        return df_novo
    except Exception as e:
        raise Exception('Arquivo "algoritmoBoW", método "tratarDataframe": \n' + str(e))


def treinarModelo(classificador, vectorizer, X_treino, rotulos_treino, textos_teste, rotulos_teste):
    try:
        # Treinando o classificador
        print(f'\nInício do treinamento {classificador.__class__.__name__} com BoW')
        classificador.fit(X_treino, rotulos_treino)
        # Salvando o classificador treinado para uso futuro
        dump(classificador, nomeArquivoClassificador)
        print(f'Fim do treinamento {classificador.__class__.__name__} com BoW\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print(f'Acurácia para {classificador.__class__.__name__}:', acuracia)

        # Calculando métricas para os resultados do TCC
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print(f'Taxa de Verdadeiros Negativos (TVB) para {classificador.__class__.__name__}:', tvb)
        print(f'Taxa de Falsos Negativos (TFD) para {classificador.__class__.__name__}:', tfd)
        print(f'Taxa de Falsos Positivos (TPF) para {classificador.__class__.__name__}:', tpf)
        print(f'F-medida para {classificador.__class__.__name__}:', f_medida)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento do modelo {classificador.__class__.__name__}: \n{str(e)}')

def treinarModelos(classificadores, vectorizer, tamanhoTeste=0.3):
    try:
        df = lerArquivo()
        textos = df[nomeColTexto].tolist()
        rotulos = df[nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador BoW
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, nomeArquivoVectorizer)

        for classificador in classificadores:
            treinarModelo(classificador, vectorizer, X_treino, rotulos_treino, textos_teste, rotulos_teste)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento dos modelos: \n{str(e)}')

def validarTextoBoW(texto):
    try:
        # Carregando o vetorizador
        vectorizer = load(nomeArquivoVectorizer)
        # Carregando o classificador treinado
        classificador = load(nomeArquivoClassificador)

        # Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)

        # Convertendo o texto para um vetor BoW
        vetor_texto = vectorizer.transform([texto_processado])

        # Realizando a predição
        predicao = classificador.predict(vetor_texto)

        return predicao <= 0.5  # Ajuste conforme necessário

    except Exception as e:
        raise Exception(f'Erro durante a validação do texto BoW: \n{str(e)}')
    
def main():
    try:
        # Criando o vetorizador BoW
        vectorizer = CountVectorizer()

        # Definindo os classificadores a serem treinados - 
        classificadores = [
            Perceptron(),
            MLPClassifier(),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            RandomForestClassifier(),
            MultinomialNB(),
            SGDClassifier(),
            PassiveAggressiveClassifier()
        ]

        # Treinando os modelos
        treinarModelos(classificadores, vectorizer)

    except Exception as e:
        raise Exception(f'Erro na função principal: \n{str(e)}')

if __name__ == '__main__':
    main() 