from geral import *

import os 
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from unidecode import unidecode

caminhoArquivos = 'arquivos/'

caminhoWord2Vec = caminhoArquivos + 'word2vec/'  # Novo caminho para salvar o modelo


nomeArquivo = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text'
nomeColTexto2 = 'text_norm'
nomeColRotulo = 'toxic'

caminhoArquivos = caminhoArquivos + 'word2vec/'
nomeArquivoModelo = caminhoArquivos + 'modelo.bin'
nomeArquivoClassificador = caminhoArquivos + 'classificador.joblib'

if not os.path.exists(caminhoWord2Vec):
    os.makedirs(caminhoWord2Vec)


def converterTextoParaVetores(texto, modelo):
    try:
        vetores = []
        for palavra in texto:
            if palavra in modelo.wv:
                vetores.append(modelo.wv[palavra])
        if vetores:
            return np.mean(vetores, axis=0)
        else:
            return np.zeros(modelo.vector_size)
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "converterTextoParaVetores": \n' + str(e))

def lerArquivo():
    try:
        df = pd.read_csv(nomeArquivo)
        df = tratarDataframe(df)
        return df
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "lerArquivo": \n' + str(e))

def tratarDataframe(df):
    try:
        print('Tamanho inicial do dataframe = ' + str(df.shape))

        #Unificando as colunas de texto e duplicando a rótulo para não perder a consistência de dados
        df_ColTexto1 = df[[nomeColTexto,  nomeColRotulo]]
        df_ColTexto2 = df[[nomeColTexto2, nomeColRotulo]]
        df_ColTexto2 = df_ColTexto2.rename(columns={nomeColTexto2: nomeColTexto})
        df_novo = pd.concat([df_ColTexto1, df_ColTexto2], ignore_index=True, axis=0, keys=None, names=[nomeColTexto, nomeColRotulo])

        #Removendo linhas com valores nulos
        print('Valores nulos = ' + str(df_novo.isnull().sum().sum()))
        df_novo = df_novo.dropna().reset_index(drop=True)

        #Substituindo caracteres acentuados por seus respectivos sem acentuação
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: unidecode(x))

        #Removendo caracteres que não sejam letras ou números
        df_novo[nomeColTexto] = df_novo[nomeColTexto].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

        print('Tamanho final do dataframe = ' + str(df_novo.shape))
        return df_novo
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "tratarDataframe": \n' + str(e))

def treinarWord2Vec(algoritmoTreinoWord2Vec=0, tamanhoTeste=0.3):
    try:
        df = lerArquivo()

        textos = [t.lower().split() for t in df[nomeColTexto]]
        rotulos = [a for a in df[nomeColRotulo]]

        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        msgAux = ' do treinamento Word2Vec utilizando algoritmo de treino ' + ('CBOW' if algoritmoTreinoWord2Vec == 0 else 'Skip-Gram')
        print('\nInício' + msgAux)
        modelo = Word2Vec(textos_treino, min_count=1, sg=algoritmoTreinoWord2Vec)
        modelo.save(nomeArquivoModelo)
        print('Fim' + msgAux + '\n')

        # Convertendo textos em vetores de palavras
        X_treino = np.array([converterTextoParaVetores(texto, modelo) for texto in textos_treino])
        X_teste = np.array([converterTextoParaVetores(texto, modelo) for texto in textos_teste])

        print('\nInício do treinamento RandomForestClassifier')
        classificador = RandomForestClassifier()
        classificador.fit(X_treino, rotulos_treino)
        dump(classificador, nomeArquivoClassificador)
        print('Fim do treinamento RandomForestClassifier\n')

        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia:', acuracia)

        # Calculando outras métricas
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb2 = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd2 = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf2 = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida2 = f1_score(rotulos_teste, predicoes)

        print('Taxa de Verdadeiros Negativos (TVB):', tvb2)
        print('Taxa de Falsos Negativos (TFD):', tfd2)
        print('Taxa de Falsos Positivos (TPF):', tpf2)
        print('F-medida:', f_medida2)
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "treinarWord2Vec": \n' + str(e))

def validarTextoWord2Vec(texto):
    try:
        modelo = Word2Vec.load(nomeArquivoModelo)
        classificador = load(nomeArquivoClassificador)

        #Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)
        vetor_texto = converterTextoParaVetores(texto_processado.split(), modelo)
        vetor_texto = np.array([vetor_texto])

        predicao = classificador.predict(vetor_texto)

        return predicao <= 0.5
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "validarTexto": \n' + str(e))
    
