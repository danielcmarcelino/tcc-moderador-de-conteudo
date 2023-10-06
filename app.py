from geral import *
from algoritmoBoW import treinarModelos, validarTextoBoW, validarTextoAdaBoost, validarTextoMultinomialNB, validarTextoSGD, validarTextoPassiveAggressive, validarTextoPerceptron, validarTextoMLP, validarTextoDecisionTree
from flask import Flask, request

app = Flask(__name__)

formatoJsonMsg = '{ "msg": "texto da mensagem aqui", "metodo": "nome_do_metodo" }'

@app.route('/')
def homepage():
    return 'API ativa. Utilize a url "/classificar" para enviar a mensagem que deseja classificar via JSON: ' + formatoJsonMsg 

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        mensagem = request.get_json()['msg'].upper()
        metodo = request.get_json()['metodo'].lower()
    except:
        return 'Ocorreu um erro ao verificar o JSON. Verifique se o mesmo está no formato correto: ' + formatoJsonMsg
    
    if mensagem == '':
        return 'A mensagem está vazia.'
    else:
        # Escolher o método desejado
        if metodo == 'bow':
            if not validarTextoBoW(mensagem):  
                return 'Mensagem possivelmente indesejável (BoW) = "' + mensagem + '"'
        elif metodo == 'adaboost':
            if not validarTextoAdaBoost(mensagem):  
                return 'Mensagem possivelmente indesejável (AdaBoost) = "' + mensagem + '"'
        elif metodo == 'multinomialnb':
            if not validarTextoMultinomialNB(mensagem):  
                return 'Mensagem possivelmente indesejável (Multinomial Naive Bayes) = "' + mensagem + '"'
        elif metodo == 'sgd':
            if not validarTextoSGD(mensagem):  
                return 'Mensagem possivelmente indesejável (SGD) = "' + mensagem + '"'
        elif metodo == 'passiveaggressive':
            if not validarTextoPassiveAggressive(mensagem):  
                return 'Mensagem possivelmente indesejável (Passive Aggressive) = "' + mensagem + '"'
        elif metodo == 'perceptron':
            if not validarTextoPerceptron(mensagem):  
                return 'Mensagem possivelmente indesejável (Perceptron) = "' + mensagem + '"'
        elif metodo == 'mlp':
            if not validarTextoMLP(mensagem):  
                return 'Mensagem possivelmente indesejável (MLPClassifier) = "' + mensagem + '"'
        elif metodo == 'decisiontree':
            if not validarTextoDecisionTree(mensagem):  
                return 'Mensagem possivelmente indesejável (DecisionTreeClassifier) = "' + mensagem + '"'
        else:
            return 'Método não reconhecido. Utilize "bow", "adaboost", "multinomialnb", "sgd", "passiveaggressive", "perceptron", "mlp" ou "decisiontree" como parâmetro.'

    return 'Mensagem OK'

if __name__ == "__main__":
    app.run(port=5000, host='localhost', debug=True)
