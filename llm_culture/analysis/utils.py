import nltk
import gensim
import spacy
import numpy as np
import ssl
import matplotlib.pyplot as plt

from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from textblob import TextBlob
import networkx as nx   

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

def display_graph(network_structure):
    pos = nx.spring_layout(network_structure)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(network_structure, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(network_structure, pos,
                           arrowstyle='->',
                           arrowsize=30)  # Increase the arrowsize value

    # labels
    nx.draw_networkx_labels(network_structure, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def PCA_plot_words(text_list):
    model = Word2Vec(text_list, min_count=1)
    model.train
    X = model.wv[model.wv.key_to_index.keys()]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.key_to_index.keys())
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()


def compute_similarity_between_texts(data_text_1, data_text_2):
    words_list_1, vectors_list_1 = data_text_1
    words_list_2, vectors_list_2 = data_text_2

    max_sims = [] 

    similarities = np.zeros((len(words_list_1), len(words_list_2)))
    for i in range(len(vectors_list_1)):
        for j in range(len(vectors_list_2)): 
            sim = get_similarity(vectors_list_1[i], vectors_list_2[j])
            similarities[i, j] = sim
        max_sim = (np.max(similarities[i, :]))
        max_sims.append(max_sim)

    return np.mean(max_sims), similarities


def extract_keywords(text, num_keywords=30):
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    
    fdist = FreqDist(filtered_tokens)
    
    keywords = [word for word, _ in fdist.most_common(num_keywords)]
    
    return keywords


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


def word_to_vector(word, model=nlp):
    return model(word).vector


def get_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0  
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    similarity = max(-1, min(1, similarity))
    
    return similarity


## Compute similarity between generations 
def compute_between_gen_similarities(similarity_matrix, n_gen, n_agents):
    between_gen_similarity_matrix = np.zeros((n_gen, n_gen))

    for i in range(n_gen):
        for j in range(n_gen):
            sim = []
            for k in range(n_agents):
                for l in range(n_agents):
                    sim.append(similarity_matrix[n_agents * i + k, n_agents * j + l])
            between_gen_similarity_matrix[i,j] = np.mean(sim)

    return between_gen_similarity_matrix


def get_polarities_subjectivities(stories):
    polarities = []
    subjectivities = []

    for gen in stories:
        pol = []
        subj = []
        for story in gen:
            story_blob = TextBlob(story)
            pol.append(story_blob.polarity)
            subj.append(story_blob.subjectivity)
        polarities.append(pol)
        subjectivities.append(subj)

    return polarities, subjectivities


# Pretty long to compute 
def compute_creativity_indexes(stories):

    def story_creativity_index(story_input):
        words_story = story_input.lower().split()
        word_vectors = [word_to_vector(word) for word in words_story]
        non_zero_word_vectors = [vector for vector in word_vectors if np.mean(vector) != 0]

        similarity_scores = []
        for vector1 in non_zero_word_vectors:
            for vector2 in non_zero_word_vectors:
                similarity = get_similarity(vector1, vector2)
                similarity_scores.append(similarity)
        return np.mean(similarity_scores)
    
    creativities = []
    for gen in stories:
        gen_creativity = []
        for story in gen:
            gen_creativity.append(story_creativity_index(story))
        creativities.append(gen_creativity)
        
    return  creativities
            
