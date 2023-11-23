from bibliotecas import *

caminhoArquivos = 'arquivos/'
nomeArquivoBD = caminhoArquivos + 'comentarios_toxicos_ptBR.csv'
nomeColTexto = 'text'
nomeColTexto2 = 'text_norm'
nomeColRotulo = 'toxic'

listaClassificadores = [
    AdaBoostClassifier(),
    DecisionTreeClassifier(),
    MLPClassifier(),
    PassiveAggressiveClassifier(),
    Perceptron(),
    RandomForestClassifier(),
    SGDClassifier()
]

def retornaAbreviacaoClassificador(classificador):
    if isinstance(classificador, AdaBoostClassifier):
        return 'ada'
    elif isinstance(classificador, DecisionTreeClassifier):
        return 'dtc'
    elif isinstance(classificador, MLPClassifier):
        return 'mlp'
    elif isinstance(classificador, PassiveAggressiveClassifier):
        return 'pac'
    elif isinstance(classificador, Perceptron):
        return 'per'
    elif isinstance(classificador, RandomForestClassifier):
        return 'rfc'
    elif isinstance(classificador, SGDClassifier):
        return 'sgd'
    else:
        return 'invalido'
    
def retornaNomeCompleto(abreviacao):
    abreviacao = abreviacao.lower()
    if abreviacao == 'ada':
        return 'AdaBoost'
    elif abreviacao == 'bow':
        return 'Bag of Words'
    elif abreviacao == 'dtc':
        return 'Decision Tree'
    elif abreviacao == 'mlp':
        return 'MLPClassifier'
    elif abreviacao == 'pac':
        return 'Passive Aggressive Classifier'
    elif abreviacao == 'per':
        return 'Perceptron'
    elif abreviacao == 'rfc':
        return 'Random Forest Classifier'
    elif abreviacao == 'sgd':
        return 'Stochastic Gradient Descent'
    elif abreviacao == 'tfi':
        return 'TF-IDF'
    elif abreviacao == 'w2v':
        return 'Word2Vec'
    else:
        return 'invalido'

def retornaNomeCompletoClassificador(classificador):
    return retornaNomeCompleto(retornaAbreviacaoClassificador(classificador))

def retornaCaminhoArquivoClassificador(caminho, classificador):
    return caminho.format(retornaAbreviacaoClassificador(classificador))

def limparTela():
    try:
        if platform.system().upper() == "WINDOWS":
            os.system("cls")
        else:
            os.system("clear")
    except Exception as e:
        raise Exception('Arquivo "geral", método "limparTela": \n' + str(e))

def criaDiretorio(caminho):
    try:
        if not os.path.exists(caminho):
            os.makedirs(caminho)
    except Exception as e:
        raise Exception('Arquivo "geral", método "criaDiretorio": \n' + str(e))
        

def lerArquivoBD():
    try:
        df = pd.read_csv(nomeArquivoBD)
        df = tratarDataframe(df)
        return df
    except Exception as e:
        raise Exception('Arquivo "geral", método "lerArquivoBD": \n' + str(e))

def removerArquivo(caminhoArquivo):
    try:
        if os.path.exists(caminhoArquivo):
            os.remove(caminhoArquivo)
    except Exception as e:
        raise Exception('Arquivo "geral", método "removerArquivo": \n' + str(e))
    
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
        raise Exception('Arquivo "geral", método "tratarDataframe": \n' + str(e))
    
def converterTextoParaVetores(tokens, modelo):
    # Função auxiliar para converter um texto em uma sequência de vetores usando um modelo Word2Vec
    vetor_resultante = []
    for token in tokens:
        if token in modelo.wv:
            vetor_resultante.append(modelo.wv[token])
    if vetor_resultante:
        return np.mean(vetor_resultante, axis=0)
    else:
        return np.zeros(modelo.vector_size)