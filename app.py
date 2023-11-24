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
            mensagem = request.get_json()['msg']
            representacao = request.get_json()['representacao']
            classificacao = request.get_json()['classificacao']
        else:
            mensagem = request.form['msg']
            representacao = request.form['representacao']
            classificacao = request.form['classificacao']

        representacao = representacao.lower()
        classificacao = classificacao.lower()

        if mensagem == '':
            return g.montaJsonRetorno(cod = 3, texto = 'Texto a ver validado é obrigatório.')
        if representacao == '' or representacao not in ['bow', 'tfi', 'w2v']:
            return g.montaJsonRetorno(cod = 4, texto = 'Algoritmo de representação inválido.')
        if classificacao == '' or classificacao not in ['ada', 'dtc', 'mlp', 'pac', 'per', 'rfc', 'sgd']:
            return g.montaJsonRetorno(cod = 5, texto = 'Algoritmo de classificação inválido.')

        mensagemOK = True

        if representacao == 'bow':
            mensagemOK = bow.validarTexto(mensagem, classificacao)
        elif representacao == 'tfi':
            mensagemOK = tfi.validarTexto(mensagem, classificacao)
        elif representacao == 'w2v':
            mensagemOK = w2v.validarTexto(mensagem, classificacao)

        resultado = 'Mensagem OK' if mensagemOK else 'Mensagem possivelmente indesejável'
        return g.montaJsonRetorno(cod = (0 if mensagemOK else 1), texto = resultado)
    except Exception as e:
        return g.montaJsonRetorno(cod = 2, texto = 'Ocorreu um erro ao validar: ' + str(e))
    
if __name__ == "__main__":
    app.run(port=5000, host='localhost', debug=True)