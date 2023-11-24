from bibliotecas import *
import geral as g

caminhoTF = g.caminhoArquivos + 'tf/'
caminhoArquivoVectorizer = caminhoTF + 'vectorizer_tfidf.joblib'
caminhoArquivoClassificador = caminhoTF + 'vectorizer_tfidf_classificador_{}.joblib'

g.criaDiretorio(caminhoTF)

def treinarModelo(classificador, vectorizer, X_treino, rotulos_treino, textos_teste, rotulos_teste):
    try:
        nomeClassificador = g.retornaNomeCompletoClassificador(classificador)
        # Treinando o classificador
        print(f'\nInício do treinamento {nomeClassificador} com TF-IDF')
        classificador.fit(X_treino, rotulos_treino)
        # Salvando o classificador treinado para uso futuro
        dump(classificador, g.retornaCaminhoArquivoClassificador(caminhoArquivoClassificador, classificador))
        print(f'Fim do treinamento {nomeClassificador} com TF-IDF\n')

        # Avaliando a acurácia no conjunto de teste
        X_teste = vectorizer.transform(textos_teste)
        predicoes = classificador.predict(X_teste)
        acuracia = accuracy_score(rotulos_teste, predicoes)
        print(f'Acurácia para {nomeClassificador}:', acuracia)

        # Calculando métricas para os resultados do TCC
        matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
        tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
        tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
        f_medida = f1_score(rotulos_teste, predicoes)

        print(f'Taxa de Verdadeiros Negativos (TVB) para {nomeClassificador}:', tvb)
        print(f'Taxa de Falsos Negativos (TFD) para {nomeClassificador}:', tfd)
        print(f'Taxa de Falsos Positivos (TPF) para {nomeClassificador}:', tpf)
        print(f'F-medida para {nomeClassificador}:', f_medida)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento do modelo {nomeClassificador}: \n{str(e)}')

def treinarModelos():
    try:
        vectorizer = TfidfVectorizer()
        tamanhoTeste=0.3

        df = g.lerArquivoBD()
        textos = df[g.nomeColTexto].tolist()
        rotulos = df[g.nomeColRotulo]

        # Dividindo o conjunto de dados em treino e teste
        textos_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=tamanhoTeste, random_state=42)

        # Criando o vetorizador TF-IDF
        X_treino = vectorizer.fit_transform(textos_treino)

        # Salvando o vetorizador para uso futuro
        dump(vectorizer, caminhoArquivoVectorizer)

        for classificador in g.listaClassificadores:
            treinarModelo(classificador, vectorizer, X_treino, rotulos_treino, textos_teste, rotulos_teste)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento dos modelos: \n{str(e)}')

def validarTexto(texto, classificador):
    try:
        # Carregando o vetorizador e o classificador treinados
        vectorizer = load(caminhoArquivoVectorizer)
        classificador = load(caminhoArquivoClassificador.format(classificador))

        # Pré-processamento do texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)

        # Vetorizar o texto
        texto_vetorizado = vectorizer.transform([texto_processado])

        # Prever a classe do texto
        predicao = classificador.predict(texto_vetorizado)

        return predicao <= 0.5

    except Exception as e:
        raise Exception(f'Erro durante a validação do texto com TF-IDF: \n{str(e)}')

def main():
    try:
        # Treinando os modelos
        treinarModelos()
    except Exception as e:
        raise Exception(f'Erro na função principal: \n{str(e)}')

if __name__ == '__main__':
    main()