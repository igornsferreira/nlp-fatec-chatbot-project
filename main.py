import spacy
from goose3 import Goose
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

nlp = spacy.load('pt_core_news_lg')

def welcome_message(user_text):
    mensagens = ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']
    for msg in mensagens:
        if msg in user_text.lower():
            return 'Olá! Sou um chatbot sobre a Seleção Brasileira de Futebol. Pergunte algo!'
    return None

def extrair_artigo():
    url = 'https://pt.wikipedia.org/wiki/Sele%C3%A7%C3%A3o_Brasileira_de_Futebol'
    g = Goose()
    article = g.extract(url)
    return [sentence for sentence in sent_tokenize(article.cleaned_text)]

original_sentences = extrair_artigo()

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

def iniciar_chat():
    print('Chatbot: Olá! Sou um chatbot sobre a Seleção Brasileira de Futebol. Pergunte algo ou digite "sair" para encerrar.')
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ['sair', 'quit', 'exit']:
            print('Chatbot: Até mais! 👋')
            break
        elif welcome_message(user_input) is not None:
            print('Chatbot:', welcome_message(user_input))
        else:
            print('Chatbot:', answer(user_input))

if __name__ == '__main__':
    iniciar_chat()