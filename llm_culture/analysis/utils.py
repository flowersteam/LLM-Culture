import json
import pickle

import nltk
import gensim
import pandas as pd
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
import os
from GRUEN.main import get_focus_score, get_grammaticality_score, get_redundancy_score, preprocess_candidates
from sentence_transformers import SentenceTransformer, util
import whylogs as why
from langkit import light_metrics, extract


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


def get_stories(folder, start_flag = None, end_flag = None):
    if start_flag != None and end_flag != None:
        pattern = rf"{start_flag}(.*?){end_flag}"
    
    json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
    all_stories = []
    # json_file  = folder + '/output.json'
    for json_file in json_files:
        with open(folder + '/' + json_file, 'r') as file:
                data = json.load(file)
                stories = data['stories']
                print("0::",len(stories))
                print("1::",len(stories[0]))
                print("2::",len(stories[0][0]))
                
                if start_flag is not None and end_flag is not None:
                    stories_gen = []
                    for gen in stories:
                        if start_flag != None and end_flag != None:
                            stories_gen.append([re.search(pattern, story).group(1).strip() for story in gen])
                        else:
                            stories_gen.append(gen)
                    all_stories.append(stories_gen)
                    print("0::",len(stories_gen))
                    print("1::",len(stories_gen[0]))
                    print("2::",len(stories_gen[0][0]))
                    #return ''
                else:
                    all_stories.append(stories)
    
    return all_stories


def get_initial_story(folder, marker = "Here is the text:"):
    json_files = [file for file in os.listdir(folder) if file.endswith('.json')]

    for json_file in json_files:
        with open(folder + '/' + json_file, 'r') as file:
            data = json.load(file)
            prompt_init = data["prompt_init"][0]
            index = prompt_init.find(marker)

            initial_story = prompt_init[index + len(marker):].strip()
            return initial_story

    

def get_plotting_infos(stories):  
    n_gen, n_agents = len(stories), len(stories[0])
    x_ticks_space = n_gen // 10 if n_gen >= 20 else 1
    return n_gen, n_agents, x_ticks_space

def preprocess_single_seed(stories):
    flat_stories = [stories[i][j] for i in range(len(stories)) for j in range(len(stories[0]))]
    keywords = [list(map(extract_keywords, s)) for s in stories]
    stem_words = [[list(map(lemmatize_stemming, keyword)) for keyword in keyword_list] for keyword_list in keywords]
    return flat_stories, keywords, stem_words

def preprocess_stories(all_seeds_stories):
    all_seeds_flat_stories = []
    all_seeds_keywords = []
    all_seeds_stem_words = []
    for stories in all_seeds_stories:
        flat_stories, keywords, stem_words = preprocess_single_seed(stories)
        all_seeds_flat_stories.append(flat_stories)
        all_seeds_keywords.append(keywords)
        all_seeds_stem_words.append(stem_words)
    
    return all_seeds_flat_stories, all_seeds_keywords, all_seeds_stem_words

def get_similarity_matrix_single_seed(flat_stories, initial_story = None):
    vect = TfidfVectorizer(min_df=1, stop_words="english", norm="l2")
    if initial_story is not None:
        flat_stories = [initial_story] + flat_stories
    tfidf = vect.fit_transform(flat_stories)                                                                                                                                                                                                                       
    similarity_matrix = tfidf * tfidf.T 
    return similarity_matrix.toarray()

def get_embeddings(stories, model = "distiluse-base-multilingual-cased-v1"):
    model = SentenceTransformer(model)
    embeddings = model.encode(stories, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def get_SBERT_similarity(story1, story2):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    embeddings1 = model.encode(story1, convert_to_tensor=True)
    embeddings2 = model.encode(story2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.cpu().numpy()

def get_similarity_matrix_single_seed_SBERT(flat_stories):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    #print(flat_stories)
    embeddings = model.encode(flat_stories, convert_to_tensor=True)
    # print(embeddings[0])
    # print(embeddings[-1])
    cosine_scores = util.cos_sim(embeddings, embeddings)
    return cosine_scores.cpu().numpy()
    similarity_matrix = embeddings @ embeddings.T / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1)[:, None])
    print(similarity_matrix)
    return similarity_matrix

def get_similarity_matrix(all_seed_flat_stories):
    all_seeds_similarity_matrix = []
    for flat_stories in all_seed_flat_stories:
        similarity_matrix = get_similarity_matrix_single_seed_SBERT(flat_stories)
        all_seeds_similarity_matrix.append(similarity_matrix)
    return all_seeds_similarity_matrix


def convert_to_json_serializable(obj):
    """
    Recursively converts non-JSON serializable objects to JSON serializable format.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj



def extract_keywords(text, num_keywords=30):
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    fdist = FreqDist(filtered_tokens)
    
    keywords = [word for word, _ in fdist.most_common(num_keywords)]
    
    return filtered_tokens
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
def compute_between_gen_similarities_single_seed(similarity_matrix, n_gen, n_agents):
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
    all_seeds_between_gen_similarity_matrix = []
    for similarity_matrix in all_seeds_similarity_matrix:
        between_gen_similarity_matrix = compute_between_gen_similarities_single_seed(similarity_matrix, n_gen, n_agents)
        all_seeds_between_gen_similarity_matrix.append(between_gen_similarity_matrix)
    return all_seeds_between_gen_similarity_matrix



def get_polarities_subjectivities_single_seed(initial_story, stories):
    polarities = []
    subjectivities = []

    initial_story_blob = TextBlob(initial_story)
    pol = []
    subj = []
    pol.append(initial_story_blob.polarity)
    subj.append(initial_story_blob.subjectivity)
    polarities.append(pol)
    subjectivities.append(subj)


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

def get_polarities_subjectivities(intial_story, all_seed_stories):
    all_seeds_polarities = []
    all_seeds_subjectivities = []
    for stories in all_seed_stories:
        polarities, subjectivities = get_polarities_subjectivities_single_seed(intial_story, stories)
        all_seeds_polarities.append(polarities)
        all_seeds_subjectivities.append(subjectivities)
    return all_seeds_polarities, all_seeds_subjectivities



# Pretty long to compute 

def get_creativity_indexes_single_seed(stories, folder, seed = 0):
    def story_creativity_index_SBERT(story_input):
        model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        #print(flat_stories)
        embeddings = model.encode(story_input.lower().split(), convert_to_tensor=True)
        # print(embeddings[0])
        # print(embeddings[-1])
        cosine_scores = util.cos_sim(embeddings, embeddings)
        return np.mean(cosine_scores.cpu().numpy())
    
    def story_creativity_index(story_input):
        words_story = story_input.lower().split()
        word_vectors = [word_to_vector(word) for word in words_story]
        non_zero_word_vectors = [vector for vector in word_vectors if np.mean(vector) != 0]

        similarity_scores = []
        for vector1 in non_zero_word_vectors:
            for vector2 in non_zero_word_vectors:
                similarity = get_similarity(vector1, vector2)
                similarity_scores.append(similarity)
        
        ## Handle the case of an empty story
        if similarity_scores:
            return np.mean(similarity_scores)
        else:
            return 0.0
    
    try:
        # Load existing data 
        file = open(f"{folder}/creativities"+str(seed)+".obj",'rb')
        creativities = pickle.load(file)
        file.close()
    except:
        # Compute creativity idx and save it 
        print(f"Computing creativity indexes of stories for {folder} dir, seed = {seed}...")
        creativities = []
        for gen in stories:
            gen_creativity = []
            for story in gen:
                gen_creativity.append(story_creativity_index_SBERT(story))
            creativities.append(gen_creativity)

        # Save the results in the texts folder
        filehandler = open(f"{folder}/creativities"+str(seed)+".obj","wb")
        pickle.dump(creativities, filehandler)
        filehandler.close()
        
    return  creativities


def get_creativity_indexes(all_seed_stories, folder):

    creativities = []
    for seed, stories in enumerate(all_seed_stories):
        creativities.append(get_creativity_indexes_single_seed(stories, folder, seed))
    return creativities




def get_grammaticality_score_single_seed(stories):
    return
    grammaticality_score = []
    for gen in stories:
        for story in gen:
            grammaticality_score.append(get_grammaticality_score([story]))
    return grammaticality_score

def get_redundancy_score_single_seed(stories):
    redundancy_score = []
    for gen in stories:
        for story in gen:
            redundancy_score.append(get_redundancy_score([story]))
    return redundancy_score

def get_focus_score_single_seed(stories):
    focus_score = []
    for gen in stories:
        for story in gen:
            focus_score.append(get_focus_score([story]))
            print(f"Focus score: {focus_score[-1]}")
    return focus_score

def get_gramm_score(all_seed_flat_stories):
    return
    all_seeds_grammaticality_score = []
    for flat_stories in all_seed_flat_stories:
        grammaticality_score = get_grammaticality_score_single_seed(flat_stories)
        all_seeds_grammaticality_score.append(grammaticality_score)
    return all_seeds_grammaticality_score

def get_red_score(all_seed_flat_stories):
    return
    all_seeds_redundancy_score = []
    print("Getting redundancy scores...")
    i = 0
    for flat_stories in all_seed_flat_stories:
        i += 1
        print(f"Story {i} / {len(all_seed_flat_stories)}")
        redundancy_score = get_redundancy_score_single_seed(flat_stories)
        all_seeds_redundancy_score.append(redundancy_score)
    print(f"redundancy:{all_seeds_redundancy_score}")
    return all_seeds_redundancy_score

def get_foc_score(all_seed_flat_stories):
    return
    print("Getting focus scores...")
    all_seeds_focus_score = []
    i = 0
    for flat_stories in all_seed_flat_stories:
        i += 1
        print(f"Seed {i} / {len(all_seed_flat_stories)}")
        focus_score = get_focus_score_single_seed(flat_stories)
        all_seeds_focus_score.append(focus_score)
    print(f"focus:{all_seeds_focus_score}")
    return all_seeds_focus_score

def get_langkit_scores_single_seed(intial_story, stories):
    flesch_reading_ease = []
    automated_readability_index = []
    aggregate_reading_level = []
    syllable_count = []
    lexicon_count = []
    sentence_count = []
    character_count = []
    letter_count = []
    polysyllable_count = []
    monosyllable_count = []
    difficult_words = []
    difficult_words_ratio = []
    polysyllable_ratio = []
    monosyllable_ratio = []
    
    stories = [[intial_story]] + stories

    for gen in stories:



        llm_schema = light_metrics.init()
        df = pd.DataFrame({'response': gen})
        enhanced_df = extract(df, schema=llm_schema)
        flesch_reading_ease.append(enhanced_df['response.flesch_reading_ease'])
        automated_readability_index.append(enhanced_df['response.automated_readability_index'])
        aggregate_reading_level.append(enhanced_df['response.aggregate_reading_level'])
        syllable_count.append(enhanced_df['response.syllable_count'])
        lexicon_count.append(enhanced_df['response.lexicon_count'])
        sentence_count.append(enhanced_df['response.sentence_count'])
        character_count.append(enhanced_df['response.character_count'])
        letter_count.append(enhanced_df['response.letter_count'])
        polysyllable_count.append(enhanced_df['response.polysyllable_count'])
        monosyllable_count.append(enhanced_df['response.monosyllable_count'])
        difficult_words.append(enhanced_df['response.difficult_words'])
        difficult_words_ratio.append(np.array(enhanced_df['response.difficult_words']) / np.array(enhanced_df['response.lexicon_count']))
        polysyllable_ratio.append(np.array(enhanced_df['response.polysyllable_count']) / np.array(enhanced_df['response.lexicon_count']))
        monosyllable_ratio.append(np.array(enhanced_df['response.monosyllable_count']) / np.array(enhanced_df['response.lexicon_count']))


    return flesch_reading_ease, automated_readability_index, aggregate_reading_level, syllable_count, lexicon_count, sentence_count, character_count, letter_count, polysyllable_count, monosyllable_count, difficult_words, difficult_words_ratio, polysyllable_ratio, monosyllable_ratio
            
    

def get_langkit_scores(intial_story, all_seed_flat_stories):
    all_seeds_flesch_reading_ease = []
    all_seeds_automated_readability_index = []
    all_seeds_aggregate_reading_level = []
    all_seeds_syllable_count = []
    all_seeds_lexicon_count = []
    all_seeds_sentence_count = []
    all_seeds_character_count = []
    all_seeds_letter_count = []
    all_seeds_polysyllable_count = []
    all_seeds_monosyllable_count = []
    all_seeds_difficult_words = []
    all_seeds_difficult_words_ratio = []
    all_seeds_polysyllable_ratio = []
    all_seeds_monosyllable_ratio = []

    for flat_stories in all_seed_flat_stories:

        flesch_reading_ease, automated_readability_index, aggregate_reading_level, syllable_count, lexicon_count, sentence_count, character_count, letter_count, polysyllable_count, monosyllable_count, difficult_words, difficult_words_ratio, monosyllable_ratio, polysyllable_ratio = get_langkit_scores_single_seed(intial_story, flat_stories)
        all_seeds_flesch_reading_ease.append(flesch_reading_ease)
        all_seeds_automated_readability_index.append(automated_readability_index)
        all_seeds_aggregate_reading_level.append(aggregate_reading_level)
        all_seeds_syllable_count.append(syllable_count)
        all_seeds_lexicon_count.append(lexicon_count)
        all_seeds_sentence_count.append(sentence_count)
        all_seeds_character_count.append(character_count)
        all_seeds_letter_count.append(letter_count)
        all_seeds_polysyllable_count.append(polysyllable_count)
        all_seeds_monosyllable_count.append(monosyllable_count)
        all_seeds_difficult_words.append(difficult_words)
        all_seeds_difficult_words_ratio.append(difficult_words_ratio)
        all_seeds_polysyllable_ratio.append(polysyllable_ratio)
        all_seeds_monosyllable_ratio.append(monosyllable_ratio)
    
    #print('all_seeds_flesch_reading_ease:', all_seeds_flesch_reading_ease)
    
    return all_seeds_flesch_reading_ease, all_seeds_automated_readability_index, all_seeds_aggregate_reading_level, all_seeds_syllable_count, all_seeds_lexicon_count, all_seeds_sentence_count, all_seeds_character_count, all_seeds_letter_count, all_seeds_polysyllable_count, all_seeds_monosyllable_count, all_seeds_difficult_words, all_seeds_difficult_words_ratio, all_seeds_polysyllable_ratio, all_seeds_monosyllable_ratio