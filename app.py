from bibliotecas import *
import geral as g
import algoritmoBoW as bow
import algoritmoWord2Vec as w2v
import algoritmoTF as tfi

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html') 

@app.route('/classificar', methods=['POST'])
def classificar():
    try:
        mensagem = ''
        representacao = ''
        classificacao = ''

        if request.content_type == 'application/json':
            mensagem = request.get_json()['msg'].upper()
            representacao = request.get_json()['representacao'].lower()
            classificacao = request.get_json()['classificacao'].lower()
        else:
            mensagem = request.form['msg'].upper()
            representacao = request.form['representacao'].lower()
            classificacao = request.form['classificacao'].lower()

        representacao = representacao.lower()
        classificacao = classificacao.lower()

        if mensagem == '':
            return jsonify({'codRetorno': 3, 'textoRetorno': 'Texto a ver validado é obrigatório.'})
        if representacao == '' or representacao not in ['bow', 'tfi', 'w2v']:
            return jsonify({'codRetorno': 4, 'textoRetorno': 'Algoritmo de representação inválido.'})
        if classificacao == '' or classificacao not in ['ada', 'dtc', 'mlp', 'pac', 'per', 'rfc', 'sgd']:
            return jsonify({'codRetorno': 5, 'textoRetorno': 'Algoritmo de classificação inválido.'})

        mensagemOK = True

        if representacao == 'bow':
            mensagemOK = bow.validarTexto(mensagem, classificacao)
        elif representacao == 'tfi':
            mensagemOK = tfi.validarTexto(mensagem, classificacao)
        elif representacao == 'w2v':
            mensagemOK = w2v.validarTexto(mensagem, classificacao)

        resultado = 'Mensagem OK' if mensagemOK else 'Mensagem possivelmente indesejável'
        return jsonify({'codRetorno': (0 if mensagemOK else 1), 'textoRetorno': resultado})
    except Exception as e:
        return jsonify({'codRetorno': 2, 'textoRetorno': 'Ocorreu um erro ao validar: ' + str(e)})
    
if __name__ == "__main__":
    app.run(port=5000, host='localhost', debug=True)