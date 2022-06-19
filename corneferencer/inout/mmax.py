import os
import shutil

from lxml import etree

import conf
from corneferencer.entities import Mention, Text


def read(inpath, clear_mentions=conf.CLEAR_INPUT):
    textname = os.path.splitext(os.path.basename(inpath))[0]
    textdir = os.path.dirname(inpath)

    mentions_path = os.path.join(textdir, '%s_mentions.xml' % textname)
    words_path = os.path.join(textdir, '%s_words.xml' % textname)

    text = Text(textname)
    text.mentions = read_mentions(mentions_path, words_path, clear_mentions)
    return text


def read_mentions(mentions_path, words_path, clear_mentions=conf.CLEAR_INPUT):
    mentions = []
    mentions_tree = etree.parse(mentions_path)
    markables = mentions_tree.xpath("//ns:markable",
                                    namespaces={'ns': 'www.eml.org/NameSpaces/mention'})
    words = get_words(words_path)

    for idx, markable in enumerate(markables):
        span = markable.attrib['span']

        dominant = ''
        if 'dominant' in markable.attrib:
            dominant = markable.attrib['dominant']

        head_orth = markable.attrib['mention_head']
        mention_words = span_to_words(span, words)

        (prec_context, follow_context, sentence,
         mnt_start_position, mnt_end_position,
         paragraph_id, sentence_id,
         first_in_sentence, first_in_paragraph) = get_context(mention_words, words)

        head = get_head(head_orth, mention_words)
        mention_group = ''
        if markable.attrib['mention_group'] != 'empty' and not clear_mentions:
            mention_group = markable.attrib['mention_group']
        mention = Mention(mnt_id=markable.attrib['id'],
                          text=span_to_text(span, words, 'orth'),
                          lemmatized_text=span_to_text(span, words, 'base'),
                          words=mention_words,
                          span=span,
                          head_orth=head_orth,
                          head=head,
                          dominant=dominant,
                          node=markable,
                          prec_context=prec_context,
                          follow_context=follow_context,
                          sentence=sentence,
                          position_in_mentions=idx,
                          start_in_words=mnt_start_position,
                          end_in_words=mnt_end_position,
                          rarest=get_rarest_word(mention_words),
                          paragraph_id=paragraph_id,
                          sentence_id=sentence_id,
                          first_in_sentence=first_in_sentence,
                          first_in_paragraph=first_in_paragraph,
                          set_id=mention_group)
        mentions.append(mention)

    return mentions


def get_words(filepath):
    tree = etree.parse(filepath)
    words = []
    for word in tree.xpath("//word"):
        hasnps = False
        if (('hasnps' in word.attrib and word.attrib['hasnps'] == 'true') or
                ('hasNps' in word.attrib and word.attrib['hasNps'] == 'true')):
            hasnps = True
        lastinsent = False
        if (('lastinsent' in word.attrib and word.attrib['lastinsent'] == 'true') or
                ('lastInSent' in word.attrib and word.attrib['lastInSent'] == 'true')):
            lastinsent = True
        lastinpar = False
        if (('lastinpar' in word.attrib and word.attrib['lastinpar'] == 'true') or
                ('lastInPar' in word.attrib and word.attrib['lastInPar'] == 'true')):
            lastinpar = True
        words.append({'id': word.attrib['id'],
                      'orth': word.text,
                      'base': word.attrib['base'],
                      'hasnps': hasnps,
                      'lastinsent': lastinsent,
                      'lastinpar': lastinpar,
                      'ctag': word.attrib['ctag'],
                      'msd': word.attrib['msd'],
                      'gender': get_gender(word.attrib['msd']),
                      'person': get_person(word.attrib['msd']),
                      'number': get_number(word.attrib['msd'])})
    return words


def span_to_words(span, words):
    fragments = span.split(',')
    mention_parts = []
    for fragment in fragments:
        mention_parts.extend(fragment_to_words(fragment, words))
    return mention_parts


def fragment_to_words(fragment, words):
    mention_parts = []
    if '..' in fragment:
        mention_parts.extend(get_multiword(fragment, words))
    else:
        mention_parts.extend(get_word(fragment, words))
    return mention_parts


def get_multiword(fragment, words):
    mention_parts = []
    boundaries = fragment.split('..')
    start_id = boundaries[0]
    end_id = boundaries[1]
    in_string = False
    for word in words:
        if word['id'] == start_id:
            in_string = True
        if in_string and not word_to_ignore(word):
            mention_parts.append(word)
        if word['id'] == end_id:
            break
    return mention_parts


def get_word(word_id, words):
    for word in words:
        if word['id'] == word_id:
            if not word_to_ignore(word):
                return [word]
            else:
                return []
    return []


def word_to_ignore(word):
    if word['ctag'] == 'interp':
        return True
    return False


def get_context(mention_words, words):
    paragraph_id = 0
    sentence_id = 0
    prec_context = []
    follow_context = []
    sentence = []
    mnt_start_position = -1
    mnt_end_position = -1
    first_word = mention_words[0]
    last_word = mention_words[-1]
    first_in_sentence = False
    first_in_paragraph = False
    for idx, word in enumerate(words):
        if word['id'] == first_word['id']:
            prec_context = get_prec_context(idx, words)
            mnt_start_position = get_mention_start(first_word, words)
            if idx == 0 or words[idx-1]['lastinsent']:
                first_in_sentence = True
            if idx == 0 or words[idx-1]['lastinpar']:
                first_in_paragraph = True
        if word['id'] == last_word['id']:
            follow_context = get_follow_context(idx, words)
            sentence = get_sentence(idx, words)
            mnt_end_position = get_mention_end(last_word, words)
            break
        if word['lastinsent']:
            sentence_id += 1
        if word['lastinpar']:
            paragraph_id += 1
    return (prec_context, follow_context, sentence, mnt_start_position, mnt_end_position,
            paragraph_id, sentence_id, first_in_sentence, first_in_paragraph)


def get_prec_context(mention_start, words):
    context = []
    context_start = mention_start - 1
    while context_start >= 0:
        if not word_to_ignore(words[context_start]):
            context.append(words[context_start])
        if len(context) == conf.CONTEXT:
            break
        context_start -= 1
    context.reverse()
    return context


def get_mention_start(first_word, words):
    start = 0
    for word in words:
        if not word_to_ignore(word):
            start += 1
        if word['id'] == first_word['id']:
            break
    return start


def get_mention_end(last_word, words):
    end = 0
    for word in words:
        if not word_to_ignore(word):
            end += 1
        if word['id'] == last_word['id']:
            break
    return end


def get_follow_context(mention_end, words):
    context = []
    context_end = mention_end + 1
    while context_end < len(words):
        if not word_to_ignore(words[context_end]):
            context.append(words[context_end])
        if len(context) == conf.CONTEXT:
            break
        context_end += 1
    return context


def get_sentence(word_idx, words):
    sentence_start = get_sentence_start(words, word_idx)
    sentence_end = get_sentence_end(words, word_idx)
    sentence = [word for word in words[sentence_start:sentence_end + 1] if not word_to_ignore(word)]
    return sentence


def get_sentence_start(words, word_idx):
    search_start = word_idx
    while word_idx >= 0:
        if words[word_idx]['lastinsent'] and search_start != word_idx:
            return word_idx + 1
        word_idx -= 1
    return 0


def get_sentence_end(words, word_idx):
    while word_idx < len(words):
        if words[word_idx]['lastinsent']:
            return word_idx
        word_idx += 1
    return len(words) - 1


def get_head(head_orth, words):
    for word in words:
        if word['orth'].lower() == head_orth.lower() or word['orth'] == head_orth:
            return word
    return None


def span_to_text(span, words, form):
    fragments = span.split(',')
    mention_parts = []
    for fragment in fragments:
        mention_parts.append(fragment_to_text(fragment, words, form))
    return u' [...] '.join(mention_parts)


def fragment_to_text(fragment, words, form):
    if '..' in fragment:
        text = get_multiword_text(fragment, words, form)
    else:
        text = get_one_word_text(fragment, words, form)
    return text


def get_multiword_text(fragment, words, form):
    mention_parts = []
    boundaries = fragment.split('..')
    start_id = boundaries[0]
    end_id = boundaries[1]
    in_string = False
    for word in words:
        if word['id'] == start_id:
            in_string = True
        if in_string and not word_to_ignore(word):
            mention_parts.append(word)
        if word['id'] == end_id:
            break
    return to_text(mention_parts, form)


def to_text(words, form):
    text = ''
    for idx, word in enumerate(words):
        if word['hasnps'] or idx == 0:
            text += word[form]
        else:
            text += u' %s' % word[form]
    return text


def get_one_word_text(word_id, words, form):
    this_word = next(word for word in words if word['id'] == word_id)
    return this_word[form]


def get_gender(msd):
    tags = msd.split(':')
    if 'm1' in tags:
        return 'm1'
    elif 'm2' in tags:
        return 'm2'
    elif 'm3' in tags:
        return 'm3'
    elif 'f' in tags:
        return 'f'
    elif 'n' in tags:
        return 'n'
    else:
        return 'unk'


def get_person(msd):
    tags = msd.split(':')
    if 'pri' in tags:
        return 'pri'
    elif 'sec' in tags:
        return 'sec'
    elif 'ter' in tags:
        return 'ter'
    else:
        return 'unk'


def get_number(msd):
    tags = msd.split(':')
    if 'sg' in tags:
        return 'sg'
    elif 'pl' in tags:
        return 'pl'
    else:
        return 'unk'


def get_rarest_word(words):
    min_freq = 0
    rarest_word = words[0]
    for i, word in enumerate(words):
        word_freq = 0
        if word['base'] in conf.FREQ_LIST:
            word_freq = conf.FREQ_LIST[word['base']]
        if i == 0 or word_freq < min_freq:
            min_freq = word_freq
            rarest_word = word
    return rarest_word


def write(inpath, outpath, text):
    textname = os.path.splitext(os.path.basename(inpath))[0]
    intextdir = os.path.dirname(inpath)
    outtextdir = os.path.dirname(outpath)

    in_mmax_path = os.path.join(intextdir, '%s.mmax' % textname)
    out_mmax_path = os.path.join(outtextdir, '%s.mmax' % textname)
    copy_mmax(in_mmax_path, out_mmax_path)

    in_words_path = os.path.join(intextdir, '%s_words.xml' % textname)
    out_words_path = os.path.join(outtextdir, '%s_words.xml' % textname)
    copy_words(in_words_path, out_words_path)

    in_mentions_path = os.path.join(intextdir, '%s_mentions.xml' % textname)
    out_mentions_path = os.path.join(outtextdir, '%s_mentions.xml' % textname)
    write_mentions(in_mentions_path, out_mentions_path, text)


def copy_mmax(src, dest):
    shutil.copyfile(src, dest)


def copy_words(src, dest):
    shutil.copyfile(src, dest)


def write_mentions(inpath, outpath, text):
    tree = etree.parse(inpath)
    mentions = tree.xpath("//ns:markable", namespaces={'ns': 'www.eml.org/NameSpaces/mention'})

    sets = text.get_sets()

    for mnt in mentions:
        mnt_set = text.get_mention_set(mnt.attrib['id'])
        if mnt_set:
            mnt.attrib['mention_group'] = mnt_set
            mnt.attrib['dominant'] = get_dominant(sets[mnt_set])
        else:
            mnt.attrib['mention_group'] = 'empty'

    with open(outpath, 'wb') as output_file:
        output_file.write(etree.tostring(tree, pretty_print=True,
                                         xml_declaration=True, encoding='UTF-8',
                                         doctype=u'<!DOCTYPE markables SYSTEM "markables.dtd">'))


def get_dominant(mentions):
    longest_mention = mentions[0]
    for mnt in mentions:
        if len(mnt.words) > len(longest_mention.words):
            longest_mention = mnt
    return longest_mention.text
