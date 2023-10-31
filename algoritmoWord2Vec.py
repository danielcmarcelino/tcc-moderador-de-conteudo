from geral import *
import numpy as np
import pandas as pd
import os
import re
from gensim.models import Word2Vec
from joblib import dump, load
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from unidecode import unidecode
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

caminhoArquivos = 'arquivos/'
caminhoWord2Vec = caminhoArquivos + 'word2vec/'
nomeArquivo = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text'
nomeColTexto2 = 'text_norm'
nomeColRotulo = 'toxic'
nomeArquivoModelo = caminhoWord2Vec + 'modelo_word2vec_skipgram.bin'
nomeArquivoClassificadorSkipGram = caminhoArquivos + 'classificador_word2vec_skipgram.joblib'

if not os.path.exists(caminhoWord2Vec):
    os.makedirs(caminhoWord2Vec)

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
        raise Exception('Arquivo "algoritmoWord2Vec", método "tratarDataframe": \n' + str(e))

def treinarModeloWord2Vec(modelo_word2vec, X_treino, rotulos_treino, textos_teste, rotulos_teste, nome_arquivo_classificador):
    try:
        # Lista de classificadores a serem treinados
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

        for classificador in classificadores:
            # Treinando o classificador
            print(f'\nInício do treinamento {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec})')
            classificador.fit(X_treino, rotulos_treino)
            # Salvando o classificador treinado para uso futuro
            dump(classificador, nome_arquivo_classificador)
            print(f'Fim do treinamento {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec})\n')

            # Avaliando a acurácia no conjunto de teste
            X_teste = [modelo_word2vec.wv[token] for token in textos_teste]
            predicoes = classificador.predict(X_teste)
            acuracia = accuracy_score(rotulos_teste, predicoes)
            print(f'Acurácia para {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec}):', acuracia)

            # Calculando métricas para os resultados do TCC
            matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
            tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
            tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
            tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
            f_medida = f1_score(rotulos_teste, predicoes)

            print(f'Taxa de Verdadeiros Negativos (TVB) para {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec}):', tvb)
            print(f'Taxa de Falsos Negativos (TFD) para {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec}):', tfd)
            print(f'Taxa de Falsos Positivos (TPF) para {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec}):', tpf)
            print(f'F-medida para {classificador.__class__.__name__} com Word2Vec ({modelo_word2vec}):', f_medida)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento dos modelos com Word2Vec: \n{str(e)}')

def validarTextoWord2Vec(texto, algoritmoClassificador):
    try:
        algoritmoClassificador = algoritmoClassificador.upper()
        modelo = Word2Vec.load(nomeArquivoModelo)
        classificador = load(caminhoArquivos + 'classificador' + algoritmoClassificador + '.joblib')

        #Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)
        vetor_texto = converterTextoParaVetores(texto_processado.split(), modelo)
        vetor_texto = np.array([vetor_texto])

        predicao = classificador.predict(vetor_texto)

        return predicao <= 0.5
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "validarTexto": \n' + str(e))

def converterTextoParaVetores(tokens, modelo):
    # Função auxiliar para converter um texto em uma sequência de vetores usando um modelo Word2Vec
    vetor_resultante = []
    for token in tokens:
        if token in modelo.wv:
            vetor_resultante.append(modelo.wv[token])
    return np.mean(vetor_resultante, axis=0)

def main():
    try:
        # Lendo o arquivo para obter o dataframe
        df = lerArquivo()

        # Treinando modelo Word2Vec com Skip-gram
        modelo_word2vec_skipgram = Word2Vec(sentences=df[nomeColTexto].apply(lambda x: x.split()), vector_size=100, window=5, sg=1, min_count=1)
        nome_arquivo_classificador_skipgram = caminhoArquivos + 'classificador_word2vec_skipgram.joblib'
        treinarModeloWord2Vec(modelo_word2vec_skipgram, df[nomeColTexto], df[nomeColRotulo], df[nomeColTexto], df[nomeColRotulo], nome_arquivo_classificador_skipgram)

        # Exemplo de uso da função validarTextoWord2Vec
        texto_para_validar = "Seu texto aqui."
        algoritmo_classificador_para_usar = "Perceptron"  # Substitua pelo classificador desejado
        resultado_validacao = validarTextoWord2Vec(texto_para_validar, algoritmo_classificador_para_usar)
        print(f"Resultado da validação: {resultado_validacao}")

    except Exception as e:
        raise Exception(f'Erro na função principal: \n{str(e)}')

if __name__ == '__main__':
    main()
