from bibliotecas import *
import geral as g

caminhoWord2Vec = g.caminhoArquivos + 'word2vec/'
caminhoArquivoVectorizer = caminhoWord2Vec + 'vectorizer_word2vec.bin'
caminhoArquivoClassificador = caminhoWord2Vec + 'vectorizer_word2vec_classificador_{}.joblib'

g.criaDiretorio(caminhoWord2Vec)

def treinarModelos():
    try:
        df = g.lerArquivoBD()

        textos = [t.lower().split() for t in df[g.nomeColTexto]]
        rotulos = [a for a in df[g.nomeColRotulo]]
        X_treino, textos_teste, rotulos_treino, rotulos_teste = train_test_split(textos, rotulos, test_size=0.3, random_state=42)
        modelo_word2vec = Word2Vec(sentences=X_treino, vector_size=100, window=5, sg=1, min_count=1)
        modelo_word2vec.save(caminhoArquivoVectorizer)

        X_treino = np.array([g.converterTextoParaVetores(texto, modelo_word2vec) for texto in X_treino])
        X_teste = np.array([g.converterTextoParaVetores(texto, modelo_word2vec) for texto in textos_teste])

        for classificador in g.listaClassificadores:
            nomeClassificador = g.retornaNomeCompletoClassificador(classificador)
            # Treinando o classificador
            print(f'\nInício do treinamento {nomeClassificador} com Word2Vec (Skip-Gram)')
            classificador.fit(X_treino, rotulos_treino)
            # Salvando o classificador treinado para uso futuro
            dump(classificador, g.retornaCaminhoArquivoClassificador(caminhoArquivoClassificador, classificador))
            print(f'Fim do treinamento {nomeClassificador} com Word2Vec (Skip-Gram)\n')

            # Avaliando a acurácia no conjunto de teste
            predicoes = classificador.predict(X_teste)
            acuracia = accuracy_score(rotulos_teste, predicoes)
            print(f'Acurácia para {nomeClassificador} com Word2Vec (Skip-Gram):', acuracia)

            # Calculando métricas para os resultados do TCC
            matriz_confusao = confusion_matrix(rotulos_teste, predicoes)
            tvb = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
            tfd = matriz_confusao[1, 0] / (matriz_confusao[1, 0] + matriz_confusao[1, 1])
            tpf = matriz_confusao[0, 1] / (matriz_confusao[0, 0] + matriz_confusao[0, 1])
            f_medida = f1_score(rotulos_teste, predicoes)

            print(f'Taxa de Verdadeiros Negativos (TVB) para {nomeClassificador} com Word2Vec (Skip-Gram):', tvb)
            print(f'Taxa de Falsos Negativos (TFD) para {nomeClassificador} com Word2Vec (Skip-Gram):', tfd)
            print(f'Taxa de Falsos Positivos (TPF) para {nomeClassificador} com Word2Vec (Skip-Gram):', tpf)
            print(f'F-medida para {nomeClassificador} com Word2Vec (Skip-Gram):', f_medida)

    except Exception as e:
        raise Exception(f'Erro durante o treinamento dos modelos com Word2Vec: \n{str(e)}')

def validarTexto(texto, classificador):
    try:
        # Carregando o vetorizador e o classificador treinados
        vetorizador = Word2Vec.load(caminhoArquivoVectorizer)
        classificador = load(caminhoArquivoClassificador.format(classificador))

        #Tratando o texto para que fique no mesmo padrão da base de dados
        texto_processado = unidecode(texto.lower())
        texto_processado = re.sub(r'[^a-zA-Z0-9\s]', '', texto_processado)
        vetor_texto = g.converterTextoParaVetores(texto_processado.split(), vetorizador)
        vetor_texto = np.array([vetor_texto])

        predicao = classificador.predict(vetor_texto)

        return predicao <= 0.5
    except Exception as e:
        raise Exception('Arquivo "algoritmoWord2Vec", método "validarTexto": \n' + str(e))

def main():
    try:
        # Treinando os modelos
        treinarModelos()
    except Exception as e:
        raise Exception(f'Erro na função principal: \n{str(e)}')

if __name__ == '__main__':
    main()
