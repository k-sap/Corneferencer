# -*- coding: utf-8 -*-

import codecs
import os
import random
import sys

from itertools import combinations
from argparse import ArgumentParser
from natsort import natsorted

sys.path.append(os.path.abspath(os.path.join('..')))

from inout import mmax, tei
from inout.constants import INPUT_FORMATS
from utils import eprint
from corneferencer.resolvers import vectors


POS_COUNT = 0
NEG_COUNT = 0


def main():
    args = parse_arguments()
    if not args.input:
        eprint("Error: Input file(s) not specified!")
    elif args.format not in INPUT_FORMATS:
        eprint("Error: Unknown input file format!")
    else:
        process_texts(args.input, args.output, args.format, args.proportion)


def parse_arguments():
    parser = ArgumentParser(description='Corneferencer: data preparator for neural nets training.')
    parser.add_argument('-i', '--input', type=str, action='store',
                        dest='input', default='',
                        help='input dir path')
    parser.add_argument('-o', '--output', type=str, action='store',
                        dest='output', default='',
                        help='output path; if not specified writes output to standard output')
    parser.add_argument('-f', '--format', type=str, action='store',
                        dest='format', default=INPUT_FORMATS[0],
                        help='input format; default: %s; possibilities: %s'
                             % (INPUT_FORMATS[0], ', '.join(INPUT_FORMATS)))
    parser.add_argument('-p', '--proportion', type=int, action='store',
                        dest='proportion', default=5,
                        help='negative examples proportion; default: 5')
    args = parser.parse_args()
    return args


def process_texts(inpath, outpath, informat, proportion):
    if os.path.isdir(inpath):
        process_directory(inpath, outpath, informat, proportion)
    else:
        eprint("Error: Specified input does not exist or is not a directory!")


def process_directory(inpath, outpath, informat, proportion):
    inpath = os.path.abspath(inpath)
    outpath = os.path.abspath(outpath)

    try:
        create_data_vectors(inpath, outpath, informat, proportion)
    finally:
        print ('Positives: ', POS_COUNT)
        print ('Negatives: ', NEG_COUNT)


def create_data_vectors(inpath, outpath, informat, proportion):
    features_file = codecs.open(outpath, 'w', 'utf-8')

    files = os.listdir(inpath)
    files = natsorted(files)

    for filename in files:
        textname = os.path.splitext(os.path.basename(filename))[0]
        textinput = os.path.join(inpath, filename)

        print ('Processing text: %s' % textname)
        text = None
        if informat == 'mmax' and filename.endswith('.mmax'):
            text = mmax.read(textinput, False)
        elif informat == 'tei':
            text = tei.read(textinput, False)

        positives, negatives = diff_mentions(text, proportion)
        write_features(features_file, positives, negatives)


def diff_mentions(text, proportion):
    sets = text.get_sets()
    all_mentions = text.get_mentions()
    positives = get_positives(sets)
    positives, negatives = get_negatives_and_update_positives(all_mentions, positives, proportion)
    return positives, negatives


def get_positives(sets):
    positives = []
    for set_id in sets:
        coref_set = sets[set_id]
        positives.extend(list(combinations(coref_set, 2)))
    return positives


def get_negatives_and_update_positives(all_mentions, positives, proportion):
    all_pairs = list(combinations(all_mentions, 2))

    all_pairs = set(all_pairs)
    negatives = [pair for pair in all_pairs if pair not in positives]
    samples_count = proportion * len(positives)
    if samples_count > len(negatives):
        samples_count = len(negatives)
        if proportion == 1:
            positives = random.sample(set(positives), samples_count)
        print (u'Więcej przypadków pozytywnych niż negatywnych!')
    negatives = random.sample(set(negatives), samples_count)
    return positives, negatives


def write_features(features_file, positives, negatives):
    global POS_COUNT
    POS_COUNT += len(positives)
    for pair in positives:
        vector = vectors.get_pair_vector(pair[0], pair[1])
        vector.append(1.0)
        features_file.write(u'%s\n' % u'\t'.join([str(feature) for feature in vector]))

    global NEG_COUNT
    NEG_COUNT += len(negatives)
    for pair in negatives:
        vector = vectors.get_pair_vector(pair[0], pair[1])
        vector.append(0.0)
        features_file.write(u'%s\n' % u'\t'.join([str(feature) for feature in vector]))


if __name__ == '__main__':
    main()
