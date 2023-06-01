import numpy as np
import os
import pandas as pd
import re
from gensim.models import Word2Vec
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from unidecode import unidecode

caminhoArquivos = 'arquivos/'

nomeArquivo = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text_norm'
nomeColRotulo = 'toxic'

caminhoArquivos = caminhoArquivos + 'word2vec/'
nomeArquivoModelo = caminhoArquivos + 'modelo.bin'
nomeArquivoClassificador = caminhoArquivos + 'classificador.joblib'

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
        raise Exception('Arquivo "Word2Vec", método "converterTextoParaVetores": \n' + str(e))
    
def lerArquivo():
    try:
        df = pd.read_csv(nomeArquivo)
        df = tratarDataframe(df)
        return df
    except Exception as e:
        raise Exception('Arquivo "Word2Vec", método "lerArquivo": \n' + str(e))

def tratarTexto(texto):
    try:
        texto = texto.strip()
        if not texto:
            return ""
        

    except Exception as e:
        raise Exception('Arquivo "Word2Vec", método "tratarTexto": \n' + str(e))

def tratarDataframe(df):
    try:
        print('Tamanho inicial do dataframe = ' + str(df.shape))

        #Removendo colunas desnecessárias
        df = df[[nomeColTexto, nomeColRotulo]] #nome drop(['text'], axis=1)

        #Removendo linhas com valores nulos
        print('Valores nulos = ' + str(df.isnull().sum().sum()))
        df = df.dropna().reset_index(drop=True)

        #Substituindo caracteres acentuados por seus respectivos sem acentuação
        df[nomeColTexto] = df[nomeColTexto].apply(lambda x: unidecode(x))

        #Removendo caracteres que não sejam letras ou números
        df[nomeColTexto] = df[nomeColTexto].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x))
        
        print('Tamanho final do dataframe = ' + str(df.shape))
        return df
    except Exception as e:
        raise Exception('Arquivo "Word2Vec", método "tratarDataframe": \n' + str(e))

def treinarWord2Vec():
    try:
        os.system('cls')

        if os.path.exists(nomeArquivoModelo):
            os.remove(nomeArquivoModelo)
        if os.path.exists(nomeArquivoClassificador):
            os.remove(nomeArquivoClassificador)

        df = lerArquivo()

        textos = [t.lower().split() for t in df[nomeColTexto]]
        rotulos = [a for a in df[nomeColRotulo]]

        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=0.30, random_state=10)

        print('\n\nInício do treinamento')
        modelo = Word2Vec(textos_treino, min_count=1)
        modelo.save(nomeArquivoModelo)
        print('Fim do treinamento\n\n')

        # Convertendo textos em vetores de palavras
        X_treino = np.array([converterTextoParaVetores(texto, modelo) for texto in textos_treino])
        X_teste = np.array([converterTextoParaVetores(texto, modelo) for texto in textos_teste])

        classifier = RandomForestClassifier()
        classifier.fit(X_treino, rotulos_treino)

        predicoes = classifier.predict(X_teste)

        acuracia = accuracy_score(rotulos_teste, predicoes)
        print('Acurácia: ', acuracia)

        dump(classifier, nomeArquivoClassificador)
        print('Modelo treinado RandomForest salvo')
    except Exception as e:
        raise Exception('Arquivo "Word2Vec", método "treinarWord2Vec": \n' + str(e))

def validarTextoWord2Vec(texto):
    try:
        modelo = Word2Vec.load(nomeArquivoModelo)
        classificador = load(nomeArquivoClassificador)

        #Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9]', '', texto_processado)
        vetor_texto = converterTextoParaVetores(texto_processado.split(), modelo)
        vetor_texto = np.array([vetor_texto])

        predicao = classificador.predict(vetor_texto)

        print(predicao[0])

        return predicao <= 0.5
    except Exception as e:
        raise Exception('Arquivo "Word2Vec", método "validarTexto": \n' + str(e))