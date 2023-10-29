from geral import *
from algoritmoBoW import treinarModelos, validarTextoBoW, validarTextoAdaBoost, validarTextoMultinomialNB, validarTextoSGD, validarTextoPassiveAggressive, validarTextoPerceptron, validarTextoMLP, validarTextoDecisionTree
from algoritmoWord2Vec import validarTextoWord2Vec
from algoritmoTF import validarTextoTFIDF
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def retornaNome(abreviacao):
    abreviacao = abreviacao.lower()

    if abreviacao == 'ada':
        return 'AdaBoost'
    elif abreviacao == 'bow':
        return 'BoW'
    elif abreviacao == 'dt':
        return 'DecisionTreeClassifier'
    elif abreviacao == 'mlp':
        return 'MLPClassifier'
    elif abreviacao == 'mnb':
        return 'Multinomial Naive Bayes'
    elif abreviacao == 'nb':
        return 'Naive Bayes'
    elif abreviacao == 'pa':
        return 'Passive Aggressive'
    elif abreviacao == 'perceptron':
        return 'Perceptron'
    elif abreviacao == 'rfc':
        return 'Random Forest Classifier'
    elif abreviacao == 'sgd':
        return 'SGD'
    elif abreviacao == 'svm':
        return 'Support Vector Machine'
    elif abreviacao == 'tfi':
        return "TF-IDF"
    elif abreviacao == 'word2vec':
        return 'Word2Vec'
    else:
        return abreviacao

@app.route('/')
def homepage():
    return render_template('index.html') 

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        mensagem = ''
        representacao = ''
        metodo = ''
        resultado = ''

        if request.content_type == 'application/json':
            mensagem = request.get_json()['msg'].upper()
            representacao = request.get_json()['representacao'].lower()
            metodo = request.get_json()['metodoClassificacao'].lower()
        else:
            mensagem = request.form['msg'].upper()
            representacao = request.form['representacao'].lower()
            metodo = request.form['metodoClassificacao'].lower()
    
        mensagemOK = True
        if mensagem == '':
            return jsonify({'textoRetorno': 'Texto a ver validado é obrigatório.'})
        elif representacao == "bow":
            # Escolher o método desejado
            if metodo == 'bow':
                mensagemOK = validarTextoBoW(mensagem)
            elif metodo == 'ada':
                mensagemOK = validarTextoAdaBoost(mensagem)
            elif metodo == 'mnb':
                mensagemOK = validarTextoMultinomialNB(mensagem)
            elif metodo == 'sgd':
                mensagemOK = validarTextoSGD(mensagem)
            elif metodo == 'pa':
                mensagemOK = validarTextoPassiveAggressive(mensagem)
            elif metodo == 'per':
                mensagemOK = validarTextoPerceptron(mensagem)
            elif metodo == 'mlp':
                mensagemOK = validarTextoMLP(mensagem)
            elif metodo == 'dt':
                mensagemOK = validarTextoDecisionTree(mensagem)
            elif metodo == 'tfi':
                mensagemOK = validarTextoTFIDF(mensagem)
        elif representacao == "word2vec" and metodo in ['rfc', 'svm', 'nb', 'ada', 'per', 'sgd', 'pa']:
            mensagemOK = validarTextoWord2Vec(mensagem, metodo)

        resultado = 'Utilizando o algoritmo de representação ' + retornaNome(representacao)
        resultado = resultado + ' com algoritmo de classificação ' + retornaNome(metodo)
        resultado = resultado + ', a mensagem "' + mensagem + '" foi classificada como '
        resultado = resultado + ('"Mensagem OK"' if mensagemOK else '"Mensagem possivelmente indesejável"')

        return jsonify({'textoRetorno': resultado})
    except Exception as e:
        return jsonify({'textoRetorno': 'Ocorreu um erro ao validar ' + str(e)})
    
if __name__ == "__main__":
    app.run(port=5000, host='localhost', debug=True)