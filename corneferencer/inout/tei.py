import gzip
import os
import shutil
from collections import defaultdict

from lxml import etree

import conf
from corneferencer.entities import Mention, Text
from corneferencer.utils import eprint

NKJP_NS = 'http://www.nkjp.pl/ns/1.0'
TEI_NS = 'http://www.tei-c.org/ns/1.0'
XI_NS = 'http://www.w3.org/2001/XInclude'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NSMAP = {None: TEI_NS,
         'nkjp': NKJP_NS,
         'xi': XI_NS}


def read(inpath, clear_mentions=conf.CLEAR_INPUT, add_single_mentions_to_cluster=True):
    textname = os.path.basename(inpath)
    print(f"tei.read {inpath}")

    text = Text(textname)

    # essential layers
    ann_segmentation = os.path.join(inpath, 'ann_segmentation.xml.gz')
    ann_morphosyntax = os.path.join(inpath, 'ann_morphosyntax.xml.gz')
    ann_mentions = os.path.join(inpath, 'ann_mentions.xml.gz')

    # additional layers
    ann_coreference = os.path.join(inpath, 'ann_coreference.xml.gz')

    if os.path.exists(ann_segmentation):
        pass
    else:
        eprint("Error: missing segmentation layer for text %s!" % textname)
        return None

    if os.path.exists(ann_morphosyntax):
        (segments, segments_ids) = read_morphosyntax(ann_morphosyntax)
    else:
        eprint("Error: missing morphosyntax layer for text %s!" % textname)
        return None

    if os.path.exists(ann_mentions):
        text.mentions = read_mentions(ann_mentions, segments, segments_ids)
    else:
        eprint("Error: missing mentions layer for text %s!" % textname)
        return None

    if os.path.exists(ann_coreference) and not clear_mentions:
        add_coreference_layer(ann_coreference, text)

    text.segments = [segment['orth'] for k, segment in segments.items()]

    mentions_sets = defaultdict(list)

    for mention in text.get_mentions():
        if mention.set:
            key = mention.set
        elif add_single_mentions_to_cluster:
            key = mention.id
        else:
            key = None
    if key:
        mentions_sets[key].append(
            (mention.start_in_words, mention.end_in_words)
        )

    text.clusters = list(mentions_sets.values())

    return text


# morphosyntax
def read_morphosyntax(ann_archive):
    segments_dict = {}
    segments_ids = []
    ann_file = gzip.open(ann_archive, 'rb')
    parser = etree.XMLParser(encoding="utf-8")
    tree = etree.parse(ann_file, parser)
    body = tree.xpath('//xmlns:body', namespaces={'xmlns': TEI_NS})[0]

    paragraphs = body.xpath(".//xmlns:p", namespaces={'xmlns': TEI_NS})
    for par in paragraphs:
        sentences = par.xpath(".//xmlns:s", namespaces={'xmlns': TEI_NS})
        for sent_id, sent in enumerate(sentences):
            segments = sent.xpath(".//xmlns:seg", namespaces={'xmlns': TEI_NS})
            for seg_id, seg in enumerate(segments):
                lastinsent = False
                lastinpar = False
                if seg_id == len(segments) - 1:
                    lastinsent = True
                    if sent_id == len(sentences) - 1:
                        lastinpar = True
                segment = read_segment(seg, lastinsent, lastinpar)
                segments_dict[segment['id']] = segment
                segments_ids.append(segment['id'])

    return segments_dict, segments_ids


def read_segment(seg, lastinsent, lastinpar):
    hasnps = False
    base = ''
    ctag = ''
    msd = ''
    orth = ''
    idx = seg.attrib['{%s}id' % XML_NS]
    for f in seg.xpath(".//xmlns:f", namespaces={'xmlns': TEI_NS}):
        if f.attrib['name'] == 'orth':
            orth = get_f_string(f)
        elif f.attrib['name'] == 'nps':
            hasnps = get_f_bin_value(f)
        elif f.attrib['name'] == 'interpretation':
            interpretation = get_f_string(f)
            (base, ctag, msd) = parse_interpretation(interpretation)
    return {'id': idx,
            'orth': orth,
            'base': base,
            'hasnps': hasnps,
            'lastinsent': lastinsent,
            'lastinpar': lastinpar,
            'ctag': ctag,
            'msd': msd,
            'number': get_number(msd),
            'person': get_person(msd),
            'gender': get_gender(msd)}


def get_f_string(f):
    return f.getchildren()[0].text


def get_f_bin_value(f):
    value = False
    if f.getchildren()[0].attrib['value'] == 'true':
        value = True
    return value


def parse_interpretation(interpretation):
    split = interpretation.split(':')
    if interpretation.startswith(':'):
        base = ':'
        ctag = 'interp'
        msd = ''
    elif len(split) > 2:
        base = split[0]
        ctag = split[1]
        msd = ':'.join(split[2:])
    else:
        base = split[0]
        ctag = split[1]
        msd = ''
    return base, ctag, msd


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


# mentions
def read_mentions(ann_archive, segments, segments_ids):
    mentions = []

    ann_file = gzip.open(ann_archive, 'rb')
    parser = etree.XMLParser(encoding="utf-8")
    tree = etree.parse(ann_file, parser)
    body = tree.xpath('//xmlns:body', namespaces={'xmlns': TEI_NS})[0]

    paragraphs = body.xpath(".//xmlns:p", namespaces={'xmlns': TEI_NS})
    mnt_id = 0
    for par_id, par in enumerate(paragraphs):
        mention_nodes = par.xpath(".//xmlns:seg", namespaces={'xmlns': TEI_NS})
        for mnt in mention_nodes:
            mnt_id += 1
            mention = get_mention(mnt, mnt_id, segments, segments_ids, par_id, sentence_id=None)
            mentions.append(mention)

    return mentions


def get_mention(mention, mnt_id, segments, segments_ids, paragraph_id, sentence_id):
    idx = mention.attrib['{%s}id' % XML_NS]

    mnt_segments = []
    for ptr in mention.xpath(".//xmlns:ptr", namespaces={'xmlns': TEI_NS}):
        seg_id = ptr.attrib['target'].split('#')[-1]
        sentence_id = int(seg_id.split('.')[-2]) if sentence_id is None else sentence_id
        if not word_to_ignore(segments[seg_id]):
            mnt_segments.append(segments[seg_id])

    semh = None
    for f in mention.xpath(".//xmlns:f", namespaces={'xmlns': TEI_NS}):
        if f.attrib['name'] == 'semh':
            semh_id = get_fval(f).split('#')[-1]
            semh = segments[semh_id]

    if len(mnt_segments) == 0:
        mnt_segments.append(semh)

    (sent_segments, prec_context, follow_context,
     first_in_sentence, first_in_paragraph) = get_context(mnt_segments, segments, segments_ids)

    mention = Mention(mnt_id=idx,
                      text=to_text(mnt_segments, 'orth'),
                      lemmatized_text=to_text(mnt_segments, 'base'),
                      words=mnt_segments,
                      span=None,
                      head_orth=semh['orth'],
                      head=semh,
                      node=mention,
                      prec_context=prec_context,
                      follow_context=follow_context,
                      sentence=sent_segments,
                      sentence_id=sentence_id,
                      paragraph_id=paragraph_id,
                      position_in_mentions=mnt_id,
                      start_in_words=segments_ids.index(mnt_segments[0]['id']),
                      end_in_words=segments_ids.index(mnt_segments[-1]['id']),
                      rarest=get_rarest_word(mnt_segments),
                      first_in_sentence=first_in_sentence,
                      first_in_paragraph=first_in_paragraph,
                      set_id=None,
                      dominant=None, )

    return mention


def get_context(mention_words, segments, segments_ids):
    prec_context = []
    follow_context = []
    sentence = []
    first_word = mention_words[0]
    last_word = mention_words[-1]
    first_in_sentence = False
    first_in_paragraph = False
    for idx, morph_id in enumerate(segments_ids):
        word = segments[morph_id]
        if word['id'] == first_word['id']:
            prec_context = get_prec_context(idx, segments, segments_ids)
            if idx == 0 or segments[segments_ids[idx - 1]]['lastinsent']:
                first_in_sentence = True
            if idx == 0 or segments[segments_ids[idx - 1]]['lastinpar']:
                first_in_paragraph = True
        if word['id'] == last_word['id']:
            follow_context = get_follow_context(idx, segments, segments_ids)
            sentence = get_sentence(idx, segments, segments_ids)
            break
    return (sentence, prec_context, follow_context, first_in_sentence, first_in_paragraph)


def get_prec_context(mention_start, segments, segments_ids):
    context = []
    context_start = mention_start - 1
    while context_start >= 0:
        if not word_to_ignore(segments[segments_ids[context_start]]):
            context.append(segments[segments_ids[context_start]])
        if len(context) == conf.CONTEXT:
            break
        context_start -= 1
    context.reverse()
    return context


def get_follow_context(mention_end, segments, segments_ids):
    context = []
    context_end = mention_end + 1
    while context_end < len(segments):
        if not word_to_ignore(segments[segments_ids[context_end]]):
            context.append(segments[segments_ids[context_end]])
        if len(context) == conf.CONTEXT:
            break
        context_end += 1
    return context


def get_sentence(word_idx, segments, segments_ids):
    sentence_start = get_sentence_start(segments, segments_ids, word_idx)
    sentence_end = get_sentence_end(segments, segments_ids, word_idx)
    sentence = [segments[morph_id] for morph_id in segments_ids[sentence_start:sentence_end + 1]
                if not word_to_ignore(segments[morph_id])]
    return sentence


def get_sentence_start(segments, segments_ids, word_idx):
    search_start = word_idx
    while word_idx >= 0:
        if segments[segments_ids[word_idx]]['lastinsent'] and search_start != word_idx:
            return word_idx + 1
        word_idx -= 1
    return 0


def get_sentence_end(segments, segments_ids, word_idx):
    while word_idx < len(segments):
        if segments[segments_ids[word_idx]]['lastinsent']:
            return word_idx
        word_idx += 1
    return len(segments) - 1


def word_to_ignore(word):
    if word['ctag'] == 'interp':
        return True
    return False


def to_text(words, form):
    text = ''
    for idx, word in enumerate(words):
        if word['hasnps'] or idx == 0:
            text += word[form]
        else:
            text += u' %s' % word[form]
    return text


def get_fval(f):
    return f.attrib['fVal']


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


# coreference
def add_coreference_layer(ann_archive, text):
    ann_file = gzip.open(ann_archive, 'rb')
    parser = etree.XMLParser(encoding="utf-8")
    tree = etree.parse(ann_file, parser)
    body = tree.xpath('//xmlns:body', namespaces={'xmlns': TEI_NS})[0]

    parts = body.xpath(".//xmlns:p", namespaces={'xmlns': TEI_NS})
    for par in parts:
        coreferences = par.xpath(".//xmlns:seg", namespaces={'xmlns': TEI_NS})
        for cor in coreferences:
            add_coreference(cor, text)


def add_coreference(coref, text):
    idx = coref.attrib['{%s}id' % XML_NS]

    coref_type = None
    dominant = None
    for f in coref.xpath(".//xmlns:f", namespaces={'xmlns': TEI_NS}):
        if f.attrib['name'] == 'type':
            coref_type = get_fval(f)
        elif f.attrib['name'] == 'dominant':
            dominant = get_fval(f)

    if coref_type == 'ident':
        for ptr in coref.xpath(".//xmlns:ptr", namespaces={'xmlns': TEI_NS}):
            mnt_id = ptr.attrib['target'].split('#')[-1]
            mention = text.get_mention(mnt_id)
            mention.set = idx
            mention.dominant = dominant


# write
def write(inpath, outpath, text):
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    for filename in os.listdir(inpath):
        if not filename.startswith('ann_coreference'):
            layer_inpath = os.path.join(inpath, filename)
            layer_outpath = os.path.join(outpath, filename)
            copy_layer(layer_inpath, layer_outpath)

    coref_outpath = os.path.join(outpath, 'ann_coreference.xml.gz')
    write_coreference(coref_outpath, text)


def copy_layer(src, dest):
    shutil.copyfile(src, dest)


def write_coreference(outpath, text):
    root, tei = write_header()
    write_body(tei, text)

    with gzip.open(outpath, 'wb') as output_file:
        output_file.write(etree.tostring(root, pretty_print=True,
                                         xml_declaration=True, encoding='UTF-8'))


def write_header():
    root = etree.Element('teiCorpus', nsmap=NSMAP)

    corpus_xinclude = etree.SubElement(root, etree.QName(XI_NS, 'include'))
    corpus_xinclude.attrib['href'] = 'PCC_header.xml'

    tei = etree.SubElement(root, 'TEI')
    tei_xinclude = etree.SubElement(tei, etree.QName(XI_NS, 'include'))
    tei_xinclude.attrib['href'] = 'header.xml'

    return root, tei


def write_body(tei, text):
    text_node = etree.SubElement(tei, 'text')
    body = etree.SubElement(text_node, 'body')
    p = etree.SubElement(body, 'p')

    sets = text.get_sets()
    for set_id in sets:
        comment_text = create_set_comment(sets[set_id])
        p.append(etree.Comment(comment_text))

        seg = etree.SubElement(p, 'seg')
        seg.attrib[etree.QName(XML_NS, 'id')] = set_id.replace('set', 'coreference')

        fs = etree.SubElement(seg, 'fs')
        fs.attrib['type'] = 'coreference'

        f_type = etree.SubElement(fs, 'f')
        f_type.attrib['name'] = 'type'
        f_type.attrib['fVal'] = 'ident'

        dominant = get_dominant(sets[set_id])
        f_dominant = etree.SubElement(fs, 'f')
        f_dominant.attrib['name'] = 'dominant'
        f_dominant.attrib['fVal'] = dominant

        for mnt in sets[set_id]:
            ptr = etree.SubElement(seg, 'ptr')
            ptr.attrib['target'] = 'ann_mentions.xml#%s' % mnt.id


def create_set_comment(mentions):
    mentions_orths = [mnt.text for mnt in mentions]
    return '  %s  ' % '; '.join(mentions_orths)


def get_dominant(mentions):
    longest_mention = mentions[0]
    for mnt in mentions:
        if len(mnt.words) > len(longest_mention.words):
            longest_mention = mnt
    return longest_mention.text
