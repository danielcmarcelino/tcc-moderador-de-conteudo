from flask import Flask, request
from Word2Vec import *

app = Flask(__name__)

formatoJsonMsg = '{ "msg": "texto da mensagem aqui" }'

@app.route('/')
def homepage():
    return 'API ativa. Utilize a url "/moderador" para enviar a mensagem que deseja classificar via JSON: ' + formatoJsonMsg 

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        mensagem = request.get_json()['msg'].upper()
    except:
        return 'Ocorreu um erro ao verificar o json. Verifique se o mesmo está no formato correto: ' + formatoJsonMsg
    
    if mensagem == '':
        return 'A mensagem está vazia.'
    else:
        if not validarTextoWord2Vec(mensagem):
            return 'Mensagem possivelmente indesejável = "' + mensagem + '"'
 
    return 'Mensagem OK'

app.run(port=5000, host='localhost', debug=True)