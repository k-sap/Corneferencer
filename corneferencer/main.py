import os
import sys
import timeit
import traceback

from argparse import ArgumentParser
from natsort import natsorted
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import conf
import utils
from inout import mmax, tei
from inout.constants import INPUT_FORMATS
from resolvers import resolve
from resolvers.constants import RESOLVERS
from utils import eprint


def main():
    args = parse_arguments()
    if not args.input:
        eprint("Error: Input file(s) not specified!")
    elif args.resolver not in RESOLVERS:
        eprint("Error: Unknown resolve algorithm!")
    elif args.format not in INPUT_FORMATS:
        eprint("Error: Unknown input file format!")
    else:
        resolver = args.resolver
        if conf.NEURAL_MODEL_ARCHITECTURE == 'siamese':
            resolver = conf.NEURAL_MODEL_ARCHITECTURE
            eprint("Warning: Using %s resolver because of selected neural model architecture!" %
                   conf.NEURAL_MODEL_ARCHITECTURE)
        process_texts(args.input, args.output, args.format, resolver, args.threshold, args.model)


def parse_arguments():
    parser = ArgumentParser(description='Corneferencer: coreference resolver using neural nets.')
    parser.add_argument('-f', '--format', type=str, action='store',
                        dest='format', default=INPUT_FORMATS[0],
                        help='input format; default: %s; possibilities: %s'
                             % (INPUT_FORMATS[0], ', '.join(INPUT_FORMATS)))
    parser.add_argument('-i', '--input', type=str, action='store',
                        dest='input', default='',
                        help='input file or dir path')
    parser.add_argument('-m', '--model', type=str, action='store',
                        dest='model', default='',
                        help='neural model path; default: %s' % conf.NEURAL_MODEL_PATH)
    parser.add_argument('-o', '--output', type=str, action='store',
                        dest='output', default='',
                        help='output path; if not specified writes output to standard output')
    parser.add_argument('-r', '--resolver', type=str, action='store',
                        dest='resolver', default=RESOLVERS[0],
                        help='resolve algorithm; default: %s; possibilities: %s'
                             % (RESOLVERS[0], ', '.join(RESOLVERS)))
    parser.add_argument('-t', '--threshold', type=float, action='store',
                        dest='threshold', default=0.85,
                        help='threshold; default: 0.85')

    args = parser.parse_args()
    return args


def process_texts(inpath, outpath, informat, resolver, threshold, model_path):
    if os.path.isdir(inpath):
        process_directory(inpath, outpath, informat, resolver, threshold, model_path)
    elif os.path.isfile(inpath):
        process_text(inpath, outpath, informat, resolver, threshold, model_path)
    else:
        eprint("Error: Specified input does not exist!")

def one_text(filename, model, inpath, outpath, resolver='all2all', informat='tei', treshold=0.85):
    textname = os.path.splitext(os.path.basename(filename))[0]
    textoutput = os.path.join(outpath, textname)
    textinput = os.path.join(inpath, filename)
    print(textinput)
    model = utils.initialize_neural_model(conf.NEURAL_MODEL_ARCHITECTURE, conf.NUMBER_OF_FEATURES, model)
    try:
        process_text(textinput, textoutput, informat, resolver, treshold, model)
    except Exception as e:
        print(textinput)
        print(e)
        traceback.print_exc()

def process_directory(inpath, outpath, informat, resolver, threshold, model):
    inpath = os.path.abspath(inpath)
    outpath = os.path.abspath(outpath)

    files = os.listdir(inpath)
    files = natsorted(files)

    # for filename in files:
    import multiprocessing
    from itertools import repeat
    pool_obj = multiprocessing.Pool(processes=1)
#    answer = pool_obj.starmap(
#        one_text, zip(files, repeat(model), repeat(inpath),
#            repeat(outpath), repeat(resolver), repeat(informat),
#            repeat(threshold))
#        )
    for p in tqdm(files):
        one_text(p, model, inpath, outpath, resolver, informat, threshold)


def process_text(inpath, outpath, informat, resolver, threshold, model):
    basename = os.path.basename(inpath)
    if informat == 'mmax' and basename.endswith('.mmax'):
        print (basename)
        text = mmax.read(inpath)
        if resolver == 'incremental':
            resolve.incremental(text, threshold, model)
        elif resolver == 'entity_based':
            resolve.entity_based(text, threshold, model)
        elif resolver == 'closest':
            resolve.closest(text, threshold, model)
        elif resolver == 'siamese':
            resolve.siamese(text, threshold, model)
        elif resolver == 'all2all':
            resolve.all2all(text, threshold, model)
        mmax.write(inpath, outpath, text)
    elif informat == 'tei':
        #print (basename)
        text = tei.read(inpath)
        #print(text)
        # print(text.get_sets())
        if resolver == 'incremental':
            resolve.incremental(text, threshold, model)
        elif resolver == 'entity_based':
            resolve.entity_based(text, threshold, model)
        elif resolver == 'closest':
            resolve.closest(text, threshold, model)
        elif resolver == 'siamese':
            resolve.siamese(text, threshold, model)
        elif resolver == 'all2all':
            resolve.all2all(text, threshold, model)
        tei.write(inpath, outpath, text)


if __name__ == '__main__':
    main()
