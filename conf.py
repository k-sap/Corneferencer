import os

import corneferencer.utils as utils

from gensim.models.word2vec import Word2Vec


CONTEXT = 5
RANDOM_WORD_VECTORS = True
CLEAR_INPUT = False
W2V_SIZE = 50
W2V_MODEL_NAME = 'w2v_allwiki_nkjpfull_50.model'

# simple or siamese
NEURAL_MODEL_ARCHITECTURE = 'simple'
NUMBER_OF_FEATURES = 1190
NEURAL_MODEL_NAME = '' # 'model_1147_features.h5'

FREQ_LIST_NAME = 'base.lst'
LEMMA2SYNONYMS_NAME = 'lemma2synonyms.map'
LEMMA2HYPERNYMS_NAME = 'lemma2hypernyms.map'
TITLE2LINKS_NAME = 'link.map'
TITLE2REDIRECT_NAME = 'redirect.map'


# do not change that
MAIN_PATH = os.path.dirname(__file__)

W2V_MODEL_PATH = os.path.join(MAIN_PATH, 'models', W2V_MODEL_NAME)
print(W2V_MODEL_PATH)
W2V_MODEL = Word2Vec.load(W2V_MODEL_PATH)

NEURAL_MODEL_PATH = os.path.join(MAIN_PATH, 'models', NEURAL_MODEL_NAME)

FREQ_LIST_PATH = os.path.join(MAIN_PATH, 'freq', FREQ_LIST_NAME)
FREQ_LIST = utils.load_freq_list(FREQ_LIST_PATH)

LEMMA2SYNONYMS_PATH = os.path.join(MAIN_PATH, 'wordnet', LEMMA2SYNONYMS_NAME)
LEMMA2SYNONYMS = utils.load_one2many_map(LEMMA2SYNONYMS_PATH)

LEMMA2HYPERNYMS_PATH = os.path.join(MAIN_PATH, 'wordnet', LEMMA2HYPERNYMS_NAME)
LEMMA2HYPERNYMS = utils.load_one2many_map(LEMMA2HYPERNYMS_PATH)

TITLE2LINKS_PATH = os.path.join(MAIN_PATH, 'wikipedia', TITLE2LINKS_NAME)
TITLE2LINKS = utils.load_one2many_map(TITLE2LINKS_PATH)

TITLE2REDIRECT_PATH = os.path.join(MAIN_PATH, 'wikipedia', TITLE2REDIRECT_NAME)
TITLE2REDIRECT = utils.load_one2one_map(TITLE2REDIRECT_PATH)
