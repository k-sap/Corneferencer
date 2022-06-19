import math
import numpy
import random
import re

import conf
from corneferencer.resolvers import constants


# mention features
def head_vec(mention):
    head_base = mention.head_orth
    if mention.head is not None:
        head_base = mention.head['base']
    return list(get_wv(conf.W2V_MODEL, head_base))


def first_word_vec(mention):
    return list(get_wv(conf.W2V_MODEL, mention.words[0]['base']))


def last_word_vec(mention):
    return list(get_wv(conf.W2V_MODEL, mention.words[-1]['base']))


def first_after_vec(mention):
    if len(mention.follow_context) > 0:
        vec = list(get_wv(conf.W2V_MODEL, mention.follow_context[0]['base']))
    else:
        vec = [0.0] * conf.W2V_SIZE
    return vec


def second_after_vec(mention):
    if len(mention.follow_context) > 1:
        vec = list(get_wv(conf.W2V_MODEL, mention.follow_context[1]['base']))
    else:
        vec = [0.0] * conf.W2V_SIZE
    return vec


def first_before_vec(mention):
    if len(mention.prec_context) > 0:
        vec = list(get_wv(conf.W2V_MODEL, mention.prec_context[-1]['base']))
    else:
        vec = [0.0] * conf.W2V_SIZE
    return vec


def second_before_vec(mention):
    if len(mention.prec_context) > 1:
        vec = list(get_wv(conf.W2V_MODEL, mention.prec_context[-2]['base']))
    else:
        vec = [0.0] * conf.W2V_SIZE
    return vec


def preceding_context_vec(mention):
    return list(get_context_vec(mention.prec_context, conf.W2V_MODEL))


def following_context_vec(mention):
    return list(get_context_vec(mention.follow_context, conf.W2V_MODEL))


def mention_vec(mention):
    return list(get_context_vec(mention.words, conf.W2V_MODEL))


def sentence_vec(mention):
    return list(get_context_vec(mention.sentence, conf.W2V_MODEL))


def mention_type(mention):
    type_vec = [0.0] * 4
    if mention.head is None:
        type_vec[3] = 1.0
    elif mention.head['ctag'] in constants.NOUN_TAGS:
        type_vec[0] = 1.0
    elif mention.head['ctag'] in constants.PPRON_TAGS:
        type_vec[1] = 1.0
    elif mention.head['ctag'] in constants.ZERO_TAGS:
        type_vec[2] = 1.0
    else:
        type_vec[3] = 1.0
    return type_vec


def is_first_second_person(mention):
    if mention.head is None:
        return 0.0
    if mention.head['person'] in constants.FIRST_SECOND_PERSON:
        return 1.0
    return 0.0


def is_demonstrative(mention):
    if mention.words[0]['base'].lower() in constants.INDICATIVE_PRONS_BASES:
        return 1.0
    return 0.0


def is_demonstrative_nominal(mention):
    if mention.head is None:
        return 0.0
    if is_demonstrative(mention) and mention.head['ctag'] in constants.NOUN_TAGS:
        return 1.0
    return 0.0


def is_demonstrative_pronoun(mention):
    if mention.head is None:
        return 0.0
    if (is_demonstrative(mention) and
            (mention.head['ctag'] in constants.PPRON_TAGS or mention.head['ctag'] in constants.ZERO_TAGS)):
        return 1.0
    return 0.0


def is_refl_pronoun(mention):
    if mention.head is None:
        return 0.0
    if mention.head['ctag'] in constants.SIEBIE_TAGS:
        return 1.0
    return 0.0


def is_first_in_sentence(mention):
    if mention.first_in_sentence:
        return 1.0
    return 0.0


def is_zero_or_pronoun(mention):
    if mention.head is None:
        return 0.0
    if mention.head['ctag'] in constants.PPRON_TAGS or mention.head['ctag'] in constants.ZERO_TAGS:
        return 1.0
    return 0.0


def head_contains_digit(mention):
    _digits = re.compile('\d')
    if _digits.search(mention.head_orth):
        return 1.0
    return 0.0


def mention_contains_digit(mention):
    _digits = re.compile('\d')
    if _digits.search(mention.text):
        return 1.0
    return 0.0


def contains_letter(mention):
    if any(c.isalpha() for c in mention.text):
        return 1.0
    return 0.0


def post_modified(mention):
    if mention.head_orth != mention.words[-1]['orth']:
        return 1.0
    return 0.0


# pair features
def distances_vec(ante, ana):
    vec = []

    mnts_intersect = pair_intersect(ante, ana)

    words_dist = [0.0] * 11
    words_bucket = 0
    if mnts_intersect != 1.0:
        words_bucket = get_distance_bucket(ana.start_in_words - ante.end_in_words)
    words_dist[words_bucket] = 1.0
    vec.extend(words_dist)

    mentions_dist = [0.0] * 11
    mentions_bucket = 0
    if mnts_intersect != 1.0:
        mentions_bucket = get_distance_bucket(ana.position_in_mentions - ante.position_in_mentions)
    if words_bucket == 10:
        mentions_bucket = 10
    mentions_dist[mentions_bucket] = 1.0
    vec.extend(mentions_dist)

    vec.append(mnts_intersect)

    return vec


def pair_intersect(ante, ana):
    for ante_word in ante.words:
        for ana_word in ana.words:
            if ana_word['id'] == ante_word['id']:
                return 1.0
    return 0.0


def head_match(ante, ana):
    if ante.head_orth.lower() == ana.head_orth.lower():
        return 1.0
    return 0.0


def exact_match(ante, ana):
    if ante.text.lower() == ana.text.lower():
        return 1.0
    return 0.0


def base_match(ante, ana):
    if ante.lemmatized_text.lower() == ana.lemmatized_text.lower():
        return 1.0
    return 0.0


def ante_contains_rarest_from_ana(ante, ana):
    ana_rarest = ana.rarest
    for word in ante.words:
        if word['base'] == ana_rarest['base']:
            return 1.0
    return 0.0


def agreement(ante, ana, tag_name):
    agr_vec = [0.0] * 3
    if (ante.head is None or ana.head is None or
            ante.head[tag_name] == 'unk' or ana.head[tag_name] == 'unk'):
        agr_vec[2] = 1.0
    elif ante.head[tag_name] == ana.head[tag_name]:
        agr_vec[0] = 1.0
    else:
        agr_vec[1] = 1.0
    return agr_vec


def is_acronym(ante, ana):
    if ana.text.upper() == ana.text:
        return check_one_way_acronym(ana.text, ante.text)
    if ante.text.upper() == ante.text:
        return check_one_way_acronym(ante.text, ana.text)
    return 0.0


def same_sentence(ante, ana):
    if ante.sentence_id == ana.sentence_id:
        return 1.0
    return 0.0


def neighbouring_sentence(ante, ana):
    if ana.sentence_id - ante.sentence_id == 1:
        return 1.0
    return 0.0


def cousin_sentence(ante, ana):
    if ana.sentence_id - ante.sentence_id == 2:
        return 1.0
    return 0.0


def distant_sentence(ante, ana):
    if ana.sentence_id - ante.sentence_id > 2:
        return 1.0
    return 0.0


def same_paragraph(ante, ana):
    if ante.paragraph_id == ana.paragraph_id:
        return 1.0
    return 0.0


def flat_gender_agreement(ante, ana):
    agr_vec = [0.0] * 3
    if (ante.head is None or ana.head is None or
            ante.head['gender'] == 'unk' or ana.head['gender'] == 'unk'):
        agr_vec[2] = 1.0
    elif (ante.head['gender'] == ana.head['gender'] or
            (ante.head['gender'] in constants.MASCULINE_TAGS and ana.head['gender'] in constants.MASCULINE_TAGS)):
        agr_vec[0] = 1.0
    else:
        agr_vec[1] = 1.0
    return agr_vec


def left_match(ante, ana):
    if (ante.text.lower().startswith(ana.text.lower()) or
            ana.text.lower().startswith(ante.text.lower())):
        return 1.0
    return 0.0


def right_match(ante, ana):
    if (ante.text.lower().endswith(ana.text.lower()) or
            ana.text.lower().endswith(ante.text.lower())):
        return 1.0
    return 0.0


def abbrev2(ante, ana):
    ante_abbrev = get_abbrev(ante)
    ana_abbrev = get_abbrev(ana)
    if ante.head_orth == ana_abbrev or ana.head_orth == ante_abbrev:
        return 1.0
    return 0.0


def string_kernel(ante, ana):
    s1 = ante.text
    s2 = ana.text
    return sk(s1, s2) / (math.sqrt(sk(s1, s1) * sk(s2, s2)))


def head_string_kernel(ante, ana):
    s1 = ante.head_orth
    s2 = ana.head_orth
    return sk(s1, s2) / (math.sqrt(sk(s1, s1) * sk(s2, s2)))


def wordnet_synonyms(ante, ana):
    ante_synonyms = set()
    if ante.head is None or ana.head is None:
        return 0.0

    if ante.head['base'] in conf.LEMMA2SYNONYMS:
        ante_synonyms = conf.LEMMA2SYNONYMS[ante.head['base']]

    ana_synonyms = set()
    if ana.head['base'] in conf.LEMMA2SYNONYMS:
        ana_synonyms = conf.LEMMA2SYNONYMS[ana.head['base']]

    if ana.head['base'] in ante_synonyms or ante.head['base'] in ana_synonyms:
        return 1.0
    return 0.0


def wordnet_ana_is_hypernym(ante, ana):
    if ante.head is None or ana.head is None:
        return 0.0

    ante_hypernyms = set()
    if ante.head['base'] in conf.LEMMA2HYPERNYMS:
        ante_hypernyms = conf.LEMMA2HYPERNYMS[ante.head['base']]

    ana_hypernyms = set()
    if ana.head['base'] in conf.LEMMA2HYPERNYMS:
        ana_hypernyms = conf.LEMMA2HYPERNYMS[ana.head['base']]

    if not ante_hypernyms or not ana_hypernyms:
        return 0.0

    if ana.head['base'] in ante_hypernyms:
        return 1.0
    return 0.0


def wordnet_ante_is_hypernym(ante, ana):
    if ante.head is None or ana.head is None:
        return 0.0

    ana_hypernyms = set()
    if ana.head['base'] in conf.LEMMA2HYPERNYMS:
        ana_hypernyms = conf.LEMMA2HYPERNYMS[ana.head['base']]

    ante_hypernyms = set()
    if ante.head['base'] in conf.LEMMA2HYPERNYMS:
        ante_hypernyms = conf.LEMMA2HYPERNYMS[ante.head['base']]

    if not ante_hypernyms or not ana_hypernyms:
        return 0.0

    if ante.head['base'] in ana_hypernyms:
        return 1.0
    return 0.0


def wikipedia_link(ante, ana):
    ante_base = ante.lemmatized_text.lower()
    ana_base = ana.lemmatized_text.lower()
    if ante_base == ana_base:
        return 1.0

    ante_links = set()
    if ante_base in conf.TITLE2LINKS:
        ante_links = conf.TITLE2LINKS[ante_base]

    ana_links = set()
    if ana_base in conf.TITLE2LINKS:
        ana_links = conf.TITLE2LINKS[ana_base]

    if ana_base in ante_links or ante_base in ana_links:
        return 1.0

    return 0.0


def wikipedia_mutual_link(ante, ana):
    ante_base = ante.lemmatized_text.lower()
    ana_base = ana.lemmatized_text.lower()
    if ante_base == ana_base:
        return 1.0

    ante_links = set()
    if ante_base in conf.TITLE2LINKS:
        ante_links = conf.TITLE2LINKS[ante_base]

    ana_links = set()
    if ana_base in conf.TITLE2LINKS:
        ana_links = conf.TITLE2LINKS[ana_base]

    if ana_base in ante_links and ante_base in ana_links:
        return 1.0

    return 0.0


def wikipedia_redirect(ante, ana):
    ante_base = ante.lemmatized_text.lower()
    ana_base = ana.lemmatized_text.lower()
    if ante_base == ana_base:
        return 1.0

    if ante_base in conf.TITLE2REDIRECT and conf.TITLE2REDIRECT[ante_base] == ana_base:
        return 1.0

    if ana_base in conf.TITLE2REDIRECT and conf.TITLE2REDIRECT[ana_base] == ante_base:
        return 1.0

    return 0.0


def samesent_anapron_antefirstinpar(ante, ana):
    if same_sentence(ante, ana) and is_zero_or_pronoun(ana) and ante.first_in_paragraph:
        return 1.0
    return 0.0


def samesent_antefirstinpar_personnumbermatch(ante, ana):
    if (same_sentence(ante, ana) and ante.first_in_paragraph
            and agreement(ante, ana, 'number')[0] and agreement(ante, ana, 'person')[0]):
        return 1.0
    return 0.0


def adjsent_anapron_adjmen_personnumbermatch(ante, ana):
    if (neighbouring_sentence(ante, ana) and is_zero_or_pronoun(ana)
            and ana.position_in_mentions - ante.position_in_mentions == 1
            and agreement(ante, ana, 'number')[0] and agreement(ante, ana, 'person')[0]):
        return 1.0
    return 0.0


def adjsent_anapron_adjmen(ante, ana):
    if (neighbouring_sentence(ante, ana) and is_zero_or_pronoun(ana)
            and ana.position_in_mentions - ante.position_in_mentions == 1):
        return 1.0
    return 0.0


# supporting functions
def get_wv(model, lemma, use_random_vec=True):
    vec = None
    if use_random_vec:
        vec = random_vec()
    try:
        vec = model.wv[lemma]
    except KeyError:
        pass
    except TypeError:
        pass
    return vec


def random_vec():
    return numpy.asarray([random.uniform(-0.25, 0.25) for i in range(0, conf.W2V_SIZE)], dtype=numpy.float32)


def get_context_vec(words, model):
    vec = numpy.zeros(conf.W2V_SIZE, dtype=numpy.float32)
    unknown_count = 0
    if len(words) != 0:
        for word in words:
            word_vec = get_wv(model, word['base'], conf.RANDOM_WORD_VECTORS)
            if word_vec is None:
                unknown_count += 1
            else:
                vec += word_vec
        significant_words = len(words) - unknown_count
        if significant_words != 0:
            vec = vec / float(significant_words)
        else:
            vec = random_vec()
    return vec


def get_distance_bucket(distance):
    if 0 <= distance <= 4:
        return distance
    elif 5 <= distance <= 7:
        return 5
    elif 8 <= distance <= 15:
        return 6
    elif 16 <= distance <= 31:
        return 7
    elif 32 <= distance <= 63:
        return 8
    elif distance >= 64:
        return 9
    return 10


def check_one_way_acronym(acronym, expression):
    initials = u''
    for expr1 in expression.split('-'):
        for expr2 in expr1.split():
            expr2 = expr2.strip()
            if expr2:
                initials += expr2[0].upper()
    if acronym == initials:
        return 1.0
    return 0.0


def get_abbrev(mention):
    abbrev = u''
    for word in mention.words:
        if word['orth'][0].isupper():
            abbrev += word['orth'][0]
    return abbrev


def sk(s1, s2):
    lam = 0.4

    p = len(s1)
    if len(s2) < len(s1):
        p = len(s2)

    h, w = len(s1)+1, len(s2)+1
    dps = [[0.0] * w for i in range(h)]
    dp = [[0.0] * w for i in range(h)]

    kernel_mat = [0.0] * (len(s1) + 1)

    for i in range(len(s1)+1):
        if i == 0:
            continue
        for j in range(len(s2)+1):
            if j == 0:
                continue
            if s1[i-1] == s2[j-1]:
                dps[i][j] = lam * lam
                kernel_mat[0] += dps[i][j]
            else:
                dps[i][j] = 0.0

    for m in range(p):
        if m == 0:
            continue

        kernel_mat[m] = 0.0
        for j in range(len(s2)+1):
            dp[m-1][j] = 0.0

        for i in range(len(s1)+1):
            dp[i][m-1] = 0.0

        for i in range(len(s1)+1):
            if i < m:
                continue
            for j in range(len(s2)+1):
                if j < m:
                    continue
                dp[i][j] = dps[i][j] + lam * dp[i - 1][j] + lam * dp[i][j - 1] - lam * lam * dp[i - 1][j - 1]

                if s1[i-1] == s2[j-1]:
                    dps[i][j] = lam * lam * dp[i - 1][j - 1]
                    kernel_mat[m] += dps[i][j]

    k = 0.0
    for i in range(p):
        k += kernel_mat[i]
    return k
