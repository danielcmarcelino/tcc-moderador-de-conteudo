import os
import platform

def limparTela():
    try:
        if platform.system().upper() == "WINDOWS":
            os.system("cls")
        else:
            os.system("clear")
    except Exception as e:
        raise Exception('Arquivo "geral", método "limparTela": \n' + str(e))

def removerArquivo(caminhoArquivo):
    try:
        if os.path.exists(caminhoArquivo):
            os.remove(caminhoArquivo)
    except Exception as e:
        raise Exception('Arquivo "geral", método "removerArquivo": \n' + str(e))