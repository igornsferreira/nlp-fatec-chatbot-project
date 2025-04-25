import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from goose3 import Goose
import nltk

nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

nlp = spacy.load('pt_core_news_lg')

def extrair_artigo():
    url = 'https://pt.wikipedia.org/wiki/Sele%C3%A7%C3%A3o_Brasileira_de_Futebol'
    g = Goose()
    article = g.extract(url)
    return [sentence for sentence in sent_tokenize(article.cleaned_text)]

original_sentences = extrair_artigo()

def welcome_message(user_text):
    mensagens = ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']
    for msg in mensagens:
        if msg in user_text.lower():
            return 'Olá! Sou um chatbot sobre a Seleção Brasileira de Futebol. Pergunte algo!'
    return None

def preprocessing(sentence):
    sentence = sentence.lower()
    tokens = [token.text for token in nlp(sentence) if not (token.is_stop or token.like_num
                                                             or token.is_punct or token.is_space
                                                             or len(token) == 1)]
    return ' '.join(tokens)

def answer(user_text, threshold=0.2):
    cleaned_sentences = [preprocessing(sentence) for sentence in original_sentences]
    user_text_cleaned = preprocessing(user_text)
    cleaned_sentences.append(user_text_cleaned)

    tfidf = TfidfVectorizer()
    x_sentences = tfidf.fit_transform(cleaned_sentences)

    similarity = cosine_similarity(x_sentences[-1], x_sentences)
    sentence_index = similarity.argsort()[0][-2] 

    if similarity[0][sentence_index] < threshold:
        return 'Desculpe! Não encontrei uma resposta adequada.'
    else:
        return original_sentences[sentence_index]

def iniciar_interface():
    def enviar(event=None):
        user_msg = entrada.get()
        if user_msg.strip():
            adicionar_mensagem(f'Você: {user_msg}', 'usuario')
            entrada.delete(0, tk.END)
            chat_text.insert(tk.END, 'Chatbot: Processando resposta...\n', 'processando')
            chat_text.see(tk.END)
            chat_text.update()

            if welcome_message(user_msg):
                resp = welcome_message(user_msg)
            else:
                resp = answer(user_msg)

            index_inicio = chat_text.search("Chatbot: Processando resposta...", "1.0", tk.END)
            if index_inicio:
                index_fim = f"{index_inicio} + 1 lines"
                chat_text.delete(index_inicio, index_fim)

            adicionar_mensagem(f'Chatbot: {resp}', 'chatbot')

    def adicionar_mensagem(texto, tipo):
        if tipo == 'usuario':
            chat_text.insert(tk.END, texto + '\n', 'usuario')
        elif tipo == 'chatbot':
            chat_text.insert(tk.END, texto + '\n\n', 'chatbot')
        chat_text.see(tk.END)

    def sair():
        janela.destroy()

    janela = tk.Tk()
    janela.title("Chatbot - Seleção Brasileira de Futebol")
    janela.configure(bg="#f0f0f0")
    janela.geometry("700x500")
    janela.minsize(500, 400)

    chat_text = tk.Text(janela, wrap=tk.WORD, bg="white", fg="black", font=("Arial", 11))
    chat_text.tag_config('usuario', foreground='blue')
    chat_text.tag_config('chatbot', foreground='green')
    chat_text.tag_config('processando', foreground='gray', font=("Arial", 10, "italic"))
    chat_text.config(state=tk.NORMAL)
    chat_text.pack(padx=10, pady=10, expand=True, fill='both')

    chat_text.insert(tk.END, "Chatbot: Olá! Pergunte algo sobre a Seleção Brasileira de Futebol.\n\n", 'chatbot')

    frame_inferior = tk.Frame(janela, bg="#f0f0f0")
    frame_inferior.pack(fill=tk.X, padx=10, pady=5)

    entrada = tk.Entry(frame_inferior, font=("Arial", 11))
    entrada.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    entrada.bind("<Return>", enviar)

    enviar_btn = tk.Button(frame_inferior, text="Enviar", command=enviar, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
    enviar_btn.pack(side=tk.LEFT, padx=(0, 5))

    sair_btn = tk.Button(frame_inferior, text="Sair", command=sair, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
    sair_btn.pack(side=tk.LEFT)

    entrada.focus()
    janela.mainloop()

if __name__ == '__main__':
    iniciar_interface()