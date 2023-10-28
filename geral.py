import os
import platform
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def limparTela():
    try:
        if platform.system().upper() == "WINDOWS":
            os.system("cls")
        else:
            os.system("clear")
    except Exception as e:
        raise Exception('Arquivo "geral", método "limparTela": \n' + str(e))


def lerArquivo(nomeArquivo):
    try:
        df = pd.read_csv(nomeArquivo)
        return df
    except Exception as e:
        raise Exception(f'Erro ao ler arquivo "{nomeArquivo}": {e}')


def removerArquivo(caminhoArquivo):
    try:
        if os.path.exists(caminhoArquivo):
            os.remove(caminhoArquivo)
    except Exception as e:
        raise Exception('Arquivo "geral", método "removerArquivo": \n' + str(e))


def remover_pontuacao(texto):
    # Remove pontuações do texto
    return texto.translate(str.maketrans("", "", string.punctuation))


def remover_stopwords(texto):
    # Remove stopwords do texto
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('portuguese'))
    palavras = texto.split()
    return ' '.join([palavra for palavra in palavras if palavra.lower() not in stop_words])


def preprocessar_texto(texto):
    # Função de pré-processamento geral
    texto = remover_pontuacao(texto)
    texto = remover_stopwords(texto)
    return texto


def criar_vectorizer_tf_idf():
    # Cria um vetorizador TF-IDF
    return TfidfVectorizer()
