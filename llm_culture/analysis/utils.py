import os
import json
# import pickle
import threading

import nltk
import gensim
# import spacy
import numpy as np
import ssl
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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


# # Download an NLP model to preprocess the text
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')
# # nlp = spacy.load('en_core_web_sm')
# nlp = None

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
_NLTK_INIT_LOCK = threading.Lock()
_NLTK_INITIALIZED = False
STOP_WORDS = None


def _ensure_nltk_ready():
    """Initialize NLTK resources once, safely, even if multiple threads race to use them."""
    global _NLTK_INITIALIZED, STOP_WORDS

    if _NLTK_INITIALIZED:
        return

    with _NLTK_INIT_LOCK:
        if _NLTK_INITIALIZED:
            return

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)

        from nltk.corpus import wordnet as _wordnet
        from nltk.corpus import stopwords as _stopwords
        _wordnet.ensure_loaded()
        STOP_WORDS = set(_stopwords.words("english"))

        _NLTK_INITIALIZED = True


def initialize_nltk():
    """Initialize the NLTK resources required by the pipeline."""
    _ensure_nltk_ready()


def lemmatize_stemming(text):
    """Lemmatize as a verb and then stem the result."""
    _ensure_nltk_ready()
    return stemmer.stem(lemmatizer.lemmatize(text, pos="v"))


def get_stories(folder):
    """Retrieve the stories from the json files in the folder

    :param folder: folder containing the json files
    :return: list of stories
    """
    _ensure_nltk_ready()
    json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
    all_stories = []
    # json_file  = folder + '/output.json'
    for json_file in json_files:
        with open(folder + '/' + json_file, 'r') as file:
                data = json.load(file)
                stories = data['stories']
                all_stories.append(stories)
    
    return all_stories
    
def get_plotting_infos(stories):  
    """Retrieve the number of generations, the number of agents and the space between x-ticks

    :param stories: list of stories
    :return: number of generations, number of agents, space between x-ticks
    """
    n_gen, n_agents = len(stories), len(stories[0])
    x_ticks_space = n_gen // 10 if n_gen >= 20 else 1
    return n_gen, n_agents, x_ticks_space

def preprocess_single_seed(stories):
    """Preprocess the stories of a single seed

    :param stories: list of stories
    :return: flat_stories, keywords, stem_words
    """
    flat_stories = [stories[i][j] for i in range(len(stories)) for j in range(len(stories[0]))]
    keywords = [list(map(extract_keywords, s)) for s in stories]
    stem_words = [[list(map(lemmatize_stemming, keyword)) for keyword in keyword_list] for keyword_list in keywords]
    return flat_stories, keywords, stem_words

def preprocess_stories(all_seeds_stories):
    """Preprocess the stories of all seeds

    :param all_seeds_stories: list of stories for each seed
    :return: all_seeds_flat_stories, all_seeds_keywords, all_seeds_stem_words
    """
    all_seeds_flat_stories = []
    all_seeds_keywords = []
    all_seeds_stem_words = []
    for stories in all_seeds_stories:
        flat_stories, keywords, stem_words = preprocess_single_seed(stories)
        all_seeds_flat_stories.append(flat_stories)
        all_seeds_keywords.append(keywords)
        all_seeds_stem_words.append(stem_words)
    
    return all_seeds_flat_stories, all_seeds_keywords, all_seeds_stem_words

def get_similarity_matrix_single_seed(flat_stories):
    """Compute the similarity matrix of the stories

    :param flat_stories: list of stories
    :return: similarity matrix
    """
    vect = TfidfVectorizer(min_df=1, stop_words="english")     
    tfidf = vect.fit_transform(flat_stories)                                                                                                                                                                                                                       
    similarity_matrix = tfidf * tfidf.T 
    return similarity_matrix.toarray()

def get_similarity_matrix(all_seed_flat_stories):
    """Compute the similarity matrix of the stories for all seeds

    :param all_seed_flat_stories: list of flat stories for each seed
    :return: list of similarity matrices
    """
    all_seeds_similarity_matrix = []
    for flat_stories in all_seed_flat_stories:
        similarity_matrix = get_similarity_matrix_single_seed(flat_stories)
        all_seeds_similarity_matrix.append(similarity_matrix)
    return all_seeds_similarity_matrix

def extract_keywords(text, num_keywords=30):
    """Extract the keywords from the text

    :param text: _description_
    :param num_keywords: _description_, defaults to 30
    :return: _description_
    """
    _ensure_nltk_ready()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS and word.isalnum()]
    fdist = FreqDist(filtered_tokens)
    
    keywords = [word for word, _ in fdist.most_common(num_keywords)]
    
    return keywords


# Tokenize and lemmatize
def preprocess(text):
    """Preprocess the text with tokenization, stop words removal and lemmatization

    :param text: text to preprocess
    :return: preprocessed text
    """
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

# def word_to_vector(word, model=nlp):
#     """Convert a word to a vector
#
#     :param word: word to convert
#     :param model: nlp model, defaults to nlp
#     :return: vector of the word
#     """
#     return model(word).vector

def get_similarity(vec1, vec2):
    """Compute the similarity between two vectors

    :param vec1: vector 1
    :param vec2: vector 2
    :return: similarity between the two vectors
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    similarity = max(-1, min(1, similarity))
    return similarity

## Compute similarity between generations 
def compute_between_gen_similarities_single_seed(similarity_matrix, n_gen, n_agents):
    """Compute the similarity between generations for a single seed

    :param similarity_matrix: similarity matrix
    :param n_gen: n_gen
    :param n_agents: n_agents
    :return: between_gen_similarity_matrix
    """
    between_gen_similarity_matrix = np.zeros((n_gen, n_gen))
    for i in range(n_gen):
        for j in range(n_gen):
            sim = []
            for k in range(n_agents):
                for l in range(n_agents):
                    sim.append(similarity_matrix[n_agents * i + k, n_agents * j + l])
            between_gen_similarity_matrix[i,j] = np.mean(sim)

    return between_gen_similarity_matrix

def compute_between_gen_similarities(all_seeds_similarity_matrix, n_gen, n_agents):
    """Compute the similarity between generations for all seeds

    :param all_seeds_similarity_matrix: list of similarity matrices
    :param n_gen: n_gen
    :param n_agents: n_agents
    :return: list of between_gen_similarity_matrix
    """
    all_seeds_between_gen_similarity_matrix = []
    for similarity_matrix in all_seeds_similarity_matrix:
        between_gen_similarity_matrix = compute_between_gen_similarities_single_seed(similarity_matrix, n_gen, n_agents)
        all_seeds_between_gen_similarity_matrix.append(between_gen_similarity_matrix)
    return all_seeds_between_gen_similarity_matrix

def get_polarities_subjectivities_single_seed(stories):
    """Compute the polarities and subjectivities of stories for a single seed.

    :param stories: list of stories for a single seed
    :return: tuple containing lists of polarities and subjectivities for each generation
    """
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

def get_polarities_subjectivities(all_seed_stories):
    """Compute the polarities and subjectivities of stories for all seeds.

    :param all_seed_stories: list of stories for each seed
    :return: tuple containing lists of polarities and subjectivities for each seed
    """
    all_seeds_polarities = []
    all_seeds_subjectivities = []
    for stories in all_seed_stories:
        polarities, subjectivities = get_polarities_subjectivities_single_seed(stories)
        all_seeds_polarities.append(polarities)
        all_seeds_subjectivities.append(subjectivities)
    return all_seeds_polarities, all_seeds_subjectivities

# # Pretty long to compute 
# def get_creativity_indexes_single_seed(stories, folder, seed=0):
#     """Compute the creativity indexes of stories for a single seed.
#
#     :param stories: list of stories for a single seed
#     :param folder: folder to save or load the creativity indexes
#     :param seed: seed number, defaults to 0
#     :return: list of creativity indexes for each generation
#     """
#     def story_creativity_index(story_input):
#         words_story = story_input.lower().split()
#         word_vectors = [word_to_vector(word) for word in words_story]
#         non_zero_word_vectors = [vector for vector in word_vectors if np.mean(vector) != 0]
#
#         similarity_scores = []
#         for vector1 in non_zero_word_vectors:
#             for vector2 in non_zero_word_vectors:
#                 similarity = get_similarity(vector1, vector2)
#                 similarity_scores.append(similarity)
#         
#         ## Handle the case of an empty story
#         if similarity_scores:
#             return np.mean(similarity_scores)
#         else:
#             return 0.0
#     try:
#         # Load existing data 
#         file = open(f"{folder}/creativities"+str(seed)+".obj",'rb')
#         creativities = pickle.load(file)
#         file.close()
#     except:
#         # Compute creativity idx and save it 
#         print(f"Computing creativity indexes of stories for {folder} dir, seed = {seed}...")
#         creativities = []
#         for gen in stories:
#             gen_creativity = []
#             for story in gen:
#                 gen_creativity.append(story_creativity_index(story))
#             creativities.append(gen_creativity)
#
#         # Save the results in the texts folder
#         filehandler = open(f"{folder}/creativities"+str(seed)+".obj","wb")
#         pickle.dump(creativities, filehandler)
#         filehandler.close()
#     return creativities
#
# def get_creativity_indexes(all_seed_stories, folder):
#     """Compute the creativity indexes of stories for all seeds.
#
#     :param all_seed_stories: list of stories for each seed
#     :param folder: folder to save or load the creativity indexes
#     :return: list of creativity indexes for each seed
#     """
#     creativities = []
#     for seed, stories in enumerate(all_seed_stories):
#         creativities.append(get_creativity_indexes_single_seed(stories, folder, seed))
#     return creativities

