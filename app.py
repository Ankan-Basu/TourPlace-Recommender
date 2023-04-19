from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from gensim.test.utils import get_tmpfile
# from gensim.models import KeyedVectors
from nltk import word_tokenize
import nltk

nltk.download('punkt')

app = Flask(__name__)
CORS(app, origins="*")

# x = ['Kolkata', 'Delhi']
df1 = pd.read_csv('./small_db.csv')

filename = './glove.6B.50d.txt'

embeddings_dict = {}
with open(filename, 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

glove_vectors = embeddings_dict

# Define a function to convert a document to a vector using both Doc2Vec and GloVe
def get_doc_vector(model, doc_words, glove_vectors):
    # Get the Doc2Vec vector
    doc_vector = model.infer_vector(doc_words)
    
    # Get the GloVe vector
    glove_vector = np.mean([glove_vectors[w] for w in doc_words if w in glove_vectors], axis=0)
    
    # If it was outside gloVe vocabulary
    try:
        if np.isnan(glove_vector):
            glove_vector = np.zeros(50)
    except ValueError:
        # if glove_vector is not 'nan' ie word exists in vocabulary
        # np.is_nan(glove_vector) will raise ValueError
        ## do nothing here
        pass
    
    # Concatenate the Doc2Vec and GloVe vectors
    return np.concatenate([doc_vector, glove_vector])


tagged_data = [TaggedDocument(words=desc.split(), tags=[i]) for i, desc in enumerate(df1['Review'])]

model1 = Doc2Vec.load('./doc2vec2.model')

# Get the document vectors
doc_vectors = [get_doc_vector(model1, word_tokenize(desc), glove_vectors) for desc in df1['Review']]


@app.route('/', methods=['GET'])
def hello():
    # print('Inside')
    query = request.args.get('q')

    tokenized_query = word_tokenize(query)

    # Calculate the cosine similarity between query and document vectors
    query_vector = get_doc_vector(model1, tokenized_query, glove_vectors)

    cosine_similarities = cosine_similarity([query_vector], doc_vectors)

    # Get the indices of the most similar documents
    most_similar_indices = np.argsort(cosine_similarities[0])[-5:][::-1]

    # Print the most similar documents
    resp = []
    for i in most_similar_indices:
        resp_item = {}
        resp_item['City'] = df1.iloc[i]['City']
        resp_item['Review'] = df1.iloc[i]['Review']
        resp.append(resp_item)
        # print(f"City: {df1.iloc[i]['City']}\nReview: {df1.iloc[i]['Review']}\n")

    
    return jsonify(resp)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
