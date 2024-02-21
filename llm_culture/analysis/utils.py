import json

import nltk
import gensim
import spacy
import numpy as np
import ssl

from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

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


def get_stories(folder):
    json_file  = folder + '/output.json'
    with open(json_file, 'r') as file:
            data = json.load(file)
            stories = data['stories']
            return stories
    

def get_plotting_infos(stories):  
    n_gen, n_agents = len(stories), len(stories[0])
    x_ticks_space = n_gen // 10 if n_gen >= 20 else 1
    return n_gen, n_agents, x_ticks_space


def preprocess_stories(stories):
    flat_stories = [stories[i][j] for i in range(len(stories)) for j in range(len(stories[0]))]
    keywords = [list(map(extract_keywords, s)) for s in stories]
    stem_words = [[list(map(lemmatize_stemming, keyword)) for keyword in keyword_list] for keyword_list in keywords]

    return flat_stories, keywords, stem_words


def get_similarity_matrix(flat_stories):
    vect = TfidfVectorizer(min_df=1, stop_words="english")     
    tfidf = vect.fit_transform(flat_stories)                                                                                                                                                                                                                       
    similarity_matrix = tfidf * tfidf.T 
    return similarity_matrix.toarray()



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
            
