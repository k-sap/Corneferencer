from corneferencer.resolvers import vectors


class Text:

    def __init__(self, text_id):
        self.__id = text_id
        self.mentions = []

    def get_mention_set(self, mnt_id):
        for mnt in self.mentions:
            if mnt.id == mnt_id:
                return mnt.set
        return None

    def get_mention(self, mnt_id):
        for mnt in self.mentions:
            if mnt.id == mnt_id:
                return mnt
        return None

    def get_mentions(self):
        return self.mentions

    def get_sets(self):
        sets = {}
        for mnt in self.mentions:
            if mnt.set and mnt.set in sets:
                sets[mnt.set].append(mnt)
            elif mnt.set:
                sets[mnt.set] = [mnt]
        return sets

    def merge_sets(self, set1, set2):
        for mnt in self.mentions:
            if mnt.set == set1:
                mnt.set = set2


class Mention:

    def __init__(self, mnt_id, text, lemmatized_text, words, span,
                 head_orth, head, dominant, node, prec_context,
                 follow_context, sentence, position_in_mentions,
                 start_in_words, end_in_words, rarest, paragraph_id, sentence_id,
                 first_in_sentence, first_in_paragraph, set_id=''):
        self.id = mnt_id
        self.set = set_id
        self.text = text
        self.lemmatized_text = lemmatized_text
        self.words = words
        self.span = span
        self.head_orth = head_orth
        self.head = head
        self.dominant = dominant
        self.node = node
        self.prec_context = prec_context
        self.follow_context = follow_context
        self.sentence = sentence
        self.position_in_mentions = position_in_mentions
        self.start_in_words = start_in_words
        self.end_in_words = end_in_words
        self.rarest = rarest
        self.paragraph_id = paragraph_id
        self.sentence_id = sentence_id
        self.first_in_sentence = first_in_sentence
        self.first_in_paragraph = first_in_paragraph
        self.features = vectors.get_mention_features(self)
