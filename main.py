# from flask import Flask, render_template, request, jsonify
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import random

# app = Flask(__name__)

# nltk.download('punkt')
# nltk.download('stopwords')

# from intents import greet,closing_phrases,thankful
# from response import greetRes,closeConversationRes,thankfulRes,fallback_responsesRes


# file_path = 'corpus.txt'
# with open(file_path, 'r') as file:
#     corpus = file.readlines()

# for line in corpus:
#     line=line.strip()  # Use strip() to remove trailing newline characters

# corpus=" . ".join(corpus)

# # print(corpus)


# def preprocess_text(text):
#     sentences = nltk.sent_tokenize(text.lower())
#     tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
#     # tokens = [[word for word in sentence if word not in string.punctuation and word not in nltk.corpus.stopwords.words('english')] for sentence in tokens]
#     return [" ".join(sentence) for sentence in tokens]

# corpus = preprocess_text(corpus)

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(corpus)

# @app.route('/')
# def index():
#     return render_template('chat.html')

# @app.route('/get_response', methods=['POST'])
# def chat():
#         user_input = request.form['user_message']
#         # print(user_input)

#         for i in greet:
#             if i.lower() in user_input.lower():
#                 return jsonify({'response': random.choice(greetRes[i])})
            
#         for i in closing_phrases:
#             if i.lower() in user_input.lower():
#                 return jsonify({'response': random.choice(closeConversationRes)})
            
#         for i in thankful:
#             if i.lower() in user_input.lower():
#                 return jsonify({'response': random.choice(thankfulRes)})
        
#         user_input = preprocess_text(user_input)

#         user_vector = vectorizer.transform(user_input)

#         similarities = cosine_similarity(user_vector, tfidf_matrix)
#         print(similarities)
        
#         similar_paragraphs = [corpus[i] for i, sim in enumerate(similarities[0]) if sim > 0.1]

#         if similar_paragraphs:
#             bot_response = "\n".join(similar_paragraphs)
#         else:
#             bot_response = random.choice(fallback_responsesRes)

#         print(bot_response)
    
#         return jsonify({'response': bot_response})
    

# if __name__ == "__main__":
#     app.run(debug=False,host='0.0.0.0')


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


nltk.download('punkt')
nltk.download('stopwords')

from intents import greet, closing_phrases, thankful
from response import greetRes, closeConversationRes, thankfulRes, fallback_responsesRes

file_path = 'corpus.txt'
with open(file_path, 'r') as file:
    corpus = file.readlines()

for line in corpus:
    line = line.strip()

corpus = " . ".join(corpus)


def preprocess_text(text):
    sentences = nltk.sent_tokenize(text.lower())
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    # tokens = [[word for word in sentence if word not in string.punctuation and word not in nltk.corpus.stopwords.words('english')] for sentence in tokens]
    return [" ".join(sentence) for sentence in tokens]

corpus = preprocess_text(corpus)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get_response")
async def chat(user_message: str = Form(...)):
    for i in greet:
        if i.lower() in user_message.lower():
            return {"response": random.choice(greetRes[i])}

    for i in closing_phrases:
        if i.lower() in user_message.lower():
            return {"response": random.choice(closeConversationRes)}

    for i in thankful:
        if i.lower() in user_message.lower():
            return {"response": random.choice(thankfulRes)}

    user_message = preprocess_text(user_message)

    user_vector = vectorizer.transform(user_message)

    similarities = cosine_similarity(user_vector, tfidf_matrix)
    # print(similarities)

    similar_paragraphs = [corpus[i] for i, sim in enumerate(similarities[0]) if sim > 0.1]

    if similar_paragraphs:
        bot_response = "\n".join(similar_paragraphs)
    else:
        bot_response = random.choice(fallback_responsesRes)

    # print(bot_response)

    return {"response": bot_response}


if __name__ == "__main__":
    import uvicorn
    from os import getenv
    port=int(getenv("PORT",8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port,reload=True)
