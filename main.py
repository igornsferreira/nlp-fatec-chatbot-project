import tkinter as tk
import concurrent.futures
import nltk
import spacy
import speech_recognition as sr

from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from goose3 import Goose
from langdetect import detect
from transformers import pipeline

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

qa_pipeline = pipeline(
    "question-answering",
    model="pierreguillou/bert-base-cased-squad-v1.1-portuguese"
)

nlp = spacy.load('pt_core_news_lg')
sentiment_analyzer = SentimentIntensityAnalyzer()

def ouvir_microfone():
    reconhecedor = sr.Recognizer()
    with sr.Microphone() as fonte_audio:
        try:
            print("Ajustando para o ruído ambiente... Aguarde.")
            reconhecedor.adjust_for_ambient_noise(fonte_audio)
            print("Pode falar!")
            audio = reconhecedor.listen(fonte_audio)
            texto = reconhecedor.recognize_google(audio, language='pt-BR')
            print("Você disse:", texto)
            return texto
        except sr.UnknownValueError:
            return "Não consegui entender o que foi dito."
        except sr.RequestError as erro:
            return f"Erro ao se conectar com o serviço de voz: {erro}"

def detectar_idioma(texto):
    try:
        return detect(texto)
    except:
        return 'unknown'

def analisar_sentimento(texto):
    scores = sentiment_analyzer.polarity_scores(texto)
    compound = scores['compound']
    if compound <= -0.4:
        return 'negativo'
    elif compound >= 0.4:
        return 'positivo'
    else:
        return 'neutro'

def extrair_artigo():
    url = 'https://pt.wikipedia.org/wiki/Sele%C3%A7%C3%A3o_Brasileira_de_Futebol'
    g = Goose()
    article = g.extract(url)
    return [sentence for sentence in sent_tokenize(article.cleaned_text)]

original_sentences_list = extrair_artigo()
original_sentences_str = " ".join(original_sentences_list)

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

def tfidf_answer(user_text, threshold=0.2):
    cleaned_sentences = [preprocessing(sentence) for sentence in original_sentences_list]
    user_text_cleaned = preprocessing(user_text)
    cleaned_sentences.append(user_text_cleaned)

    tfidf = TfidfVectorizer()
    x_sentences = tfidf.fit_transform(cleaned_sentences)

    similarity = cosine_similarity(x_sentences[-1], x_sentences)
    sentence_index = similarity.argsort()[0][-2]

    if similarity[0][sentence_index] < threshold:
        return 'Desculpe! Não encontrei uma resposta adequada.'
    else:
        return original_sentences_list[sentence_index]

def huggingface_answer(user_text):
    return qa_pipeline(question=user_text, context=original_sentences_str)['answer']

def answer(user_text, threshold=0.2, timeout=120):
    idioma = detectar_idioma(user_text)

    if idioma != 'pt':
        return "Detectei que você está usando outro idioma. Por favor, conduza a conversa em português (PT-BR)."

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(huggingface_answer, user_text)
            resp = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print("[Fallback] Tempo esgotado com HuggingFace. Usando TF-IDF.")
        resp = tfidf_answer(user_text, threshold)
    except Exception as e:
        print(f"[Fallback] Erro com HuggingFace: {e}. Usando TF-IDF.")
        resp = tfidf_answer(user_text, threshold)

    sentimento = analisar_sentimento(user_text)
    if sentimento == 'negativo':
        resp += "\n\nPercebi que você está um pouco chateado. Lembre-se: a Seleção Brasileira tem uma linda história cheia de conquistas. Pergunte mais coisas sobre ela!"

    return resp

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

    def falar():
        chat_text.insert(tk.END, 'Chatbot: Aguardando sua fala...\n', 'processando')
        chat_text.see(tk.END)
        chat_text.update()

        user_msg = ouvir_microfone()

        if user_msg:
            index_inicio = chat_text.search("Chatbot: Aguardando sua fala...", "1.0", tk.END)
            if index_inicio:
                index_fim = f"{index_inicio} + 1 lines"
                chat_text.delete(index_inicio, index_fim)

            adicionar_mensagem(f'Você (voz): {user_msg}', 'usuario')
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

    def sair():
        janela.destroy()

    def on_enter_button(event):
        event.widget.config(bg="#45a049")

    def on_leave_button(event):
        event.widget.config(bg="#4CAF50")

    def on_enter_voice_button(event):
        event.widget.config(bg="#1976D2")

    def on_leave_voice_button(event):
        event.widget.config(bg="#2196F3")

    def on_enter_exit_button(event):
        event.widget.config(bg="#d32f2f")

    def on_leave_exit_button(event):
        event.widget.config(bg="#f44336")

    janela = tk.Tk()
    janela.title("Chatbot - Seleção Brasileira de Futebol")
    janela.configure(bg="#ffffff")
    janela.geometry("800x600")
    janela.minsize(600, 450)

    header_frame = tk.Frame(janela, bg="#009739", height=60)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)

    title_label = tk.Label(header_frame, text="CHATBOT SELEÇÃO BRASILEIRA", 
                          bg="#009739", fg="white", 
                          font=("Arial", 16, "bold"))
    title_label.pack(expand=True)

    main_frame = tk.Frame(janela, bg="#f5f5f5")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    chat_frame = tk.Frame(main_frame, bg="#ffffff", relief=tk.RAISED, bd=2)
    chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    scrollbar = tk.Scrollbar(chat_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chat_text = tk.Text(chat_frame, 
                       wrap=tk.WORD, 
                       bg="#ffffff", 
                       fg="#333333", 
                       font=("Segoe UI", 11),
                       relief=tk.FLAT,
                       bd=0,
                       padx=15,
                       pady=15,
                       selectbackground="#e3f2fd",
                       yscrollcommand=scrollbar.set)
    
    scrollbar.config(command=chat_text.yview)
    
    chat_text.tag_config('usuario', 
                        foreground='#1976D2', 
                        font=("Segoe UI", 11, "bold"))
    chat_text.tag_config('chatbot', 
                        foreground='#2E7D32', 
                        font=("Segoe UI", 11))
    chat_text.tag_config('processando', 
                        foreground='#757575', 
                        font=("Segoe UI", 10, "italic"))
    
    chat_text.config(state=tk.NORMAL)
    chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    welcome_text = """Chatbot: Olá! Bem-vindo ao assistente da Seleção Brasileira!

Eu posso responder perguntas sobre a Seleção Brasileira de Futebol, seus jogadores, conquistas e muito mais!

Digite ou fale sua pergunta!Utilize o botão "Falar" para enviar sua pergunta por voz.

"""
    chat_text.insert(tk.END, welcome_text, 'chatbot')

    bottom_frame = tk.Frame(main_frame, bg="#f5f5f5")
    bottom_frame.pack(fill=tk.X)

    input_container = tk.Frame(bottom_frame, bg="#ffffff", relief=tk.SOLID, bd=1)
    input_container.pack(fill=tk.X, pady=(0, 10))

    entrada = tk.Entry(input_container, 
                      font=("Segoe UI", 12),
                      bg="#ffffff",
                      fg="#333333",
                      relief=tk.FLAT,
                      bd=0)
    entrada.pack(fill=tk.X, padx=15, pady=12)
    entrada.bind("<Return>", enviar)

    buttons_frame = tk.Frame(bottom_frame, bg="#f5f5f5")
    buttons_frame.pack(fill=tk.X)

    enviar_btn = tk.Button(buttons_frame, 
                          text="Enviar", 
                          command=enviar, 
                          bg="#4CAF50", 
                          fg="white", 
                          font=("Segoe UI", 11, "bold"),
                          relief=tk.FLAT,
                          bd=0,
                          padx=25,
                          pady=10,
                          cursor="hand2")
    enviar_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    enviar_btn.bind("<Enter>", on_enter_button)
    enviar_btn.bind("<Leave>", on_leave_button)

    falar_btn = tk.Button(buttons_frame, 
                         text="Falar", 
                         command=falar, 
                         bg="#2196F3", 
                         fg="white", 
                         font=("Segoe UI", 11, "bold"),
                         relief=tk.FLAT,
                         bd=0,
                         padx=25,
                         pady=10,
                         cursor="hand2")
    falar_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    falar_btn.bind("<Enter>", on_enter_voice_button)
    falar_btn.bind("<Leave>", on_leave_voice_button)

    def limpar_chat():
        chat_text.delete(1.0, tk.END)
        chat_text.insert(tk.END, "Chatbot: Chat limpo! Faça uma nova pergunta sobre a Seleção Brasileira!\n\n", 'chatbot')

    limpar_btn = tk.Button(buttons_frame, 
                          text="Limpar", 
                          command=limpar_chat, 
                          bg="#FF9800", 
                          fg="white", 
                          font=("Segoe UI", 11, "bold"),
                          relief=tk.FLAT,
                          bd=0,
                          padx=25,
                          pady=10,
                          cursor="hand2")
    limpar_btn.pack(side=tk.LEFT, padx=(0, 10))

    sair_btn = tk.Button(buttons_frame, 
                        text="Sair", 
                        command=sair, 
                        bg="#f44336", 
                        fg="white", 
                        font=("Segoe UI", 11, "bold"),
                        relief=tk.FLAT,
                        bd=0,
                        padx=25,
                        pady=10,
                        cursor="hand2")
    sair_btn.pack(side=tk.RIGHT)
    
    sair_btn.bind("<Enter>", on_enter_exit_button)
    sair_btn.bind("<Leave>", on_leave_exit_button)

    footer_frame = tk.Frame(janela, bg="#009739", height=30)
    footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
    footer_frame.pack_propagate(False)

    entrada.focus()
    
    janela.update_idletasks()
    width = janela.winfo_width()
    height = janela.winfo_height()
    x = (janela.winfo_screenwidth() // 2) - (width // 2)
    y = (janela.winfo_screenheight() // 2) - (height // 2)
    janela.geometry(f'{width}x{height}+{x}+{y}')

    janela.mainloop()

if __name__ == '__main__':
    iniciar_interface()