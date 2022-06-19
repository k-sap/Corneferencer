from __future__ import print_function

import codecs
import sys

import javaobj

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda
from tensorflow.keras import backend as K


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def initialize_neural_model(architecture, number_of_features, path_to_model):
    model = None
    if architecture == 'simple':
        model = initialize_simple_model(number_of_features, path_to_model)
    elif architecture == 'siamese':
        model = initialize_siamese_model(number_of_features, path_to_model)
    return model


def initialize_simple_model(number_of_features, path_to_model):
    inputs = Input(shape=(number_of_features,))

    output_from_1st_layer = Dense(1000)(inputs)
    output_from_1st_layer = BatchNormalization()(output_from_1st_layer)
    output_from_1st_layer = Activation('relu')(output_from_1st_layer)
    output_from_1st_layer = Dropout(0.2)(output_from_1st_layer)

    output_from_2nd_layer = Dense(500)(output_from_1st_layer)
    output_from_2nd_layer = BatchNormalization()(output_from_2nd_layer)
    output_from_2nd_layer = Activation('relu')(output_from_2nd_layer)
    output_from_2nd_layer = Dropout(0.2)(output_from_2nd_layer)

    output_from_3rd_layer = Dense(300)(output_from_2nd_layer)
    output_from_3rd_layer = BatchNormalization()(output_from_3rd_layer)
    output_from_3rd_layer = Activation('relu')(output_from_3rd_layer)
    output_from_3rd_layer = Dropout(0.2)(output_from_3rd_layer)

    output = Dense(1, activation='sigmoid')(output_from_3rd_layer)

    model = Model(inputs, output)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(path_to_model)
    model.load_weights(path_to_model)

    return model


def initialize_siamese_model(number_of_features, path_to_model):
    input_dim = number_of_features

    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    model.compile(loss=contrastive_loss, optimizer='Adam')
    model.load_weights(path_to_model)

    return model


def create_base_network(input_dim):
    seq = Sequential()

    seq.add(Dense(1000, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(BatchNormalization())

    seq.add(Dense(500, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(BatchNormalization())

    seq.add(Dense(300, activation='relu'))
    return seq


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def load_freq_list(freq_path):
    freq_list = {}
    with codecs.open(freq_path, 'r', 'utf-8') as freq_file:
        lines = freq_file.readlines()
        for line in lines:
            line_parts = line.split()
            freq = int(line_parts[0])
            base = line_parts[1]
            if base not in freq_list:
                freq_list[base] = freq
    return freq_list


def load_one2many_map(map_path):
    this_map = {}
    marshaller = javaobj.JavaObjectUnmarshaller(open(map_path, 'rb'))
    pobj = marshaller.readObject()
    jmap_annotations = pobj.__dict__['annotations']
    jmap_annotations_count = len(jmap_annotations)
    for i in range(jmap_annotations_count):
        if i % 2 == 1:
            mapped_elements = set(jmap_annotations[i+1].__dict__['annotations'])
            this_map[jmap_annotations[i]] = mapped_elements
    return this_map


def load_one2one_map(map_path):
    this_map = {}
    marshaller = javaobj.JavaObjectUnmarshaller(open(map_path, 'rb'))
    pobj = marshaller.readObject()
    jmap_annotations = pobj.__dict__['annotations']
    jmap_annotations_count = len(jmap_annotations)
    for i in range(jmap_annotations_count):
        if i % 2 == 1:
            element = jmap_annotations[i+1]
            this_map[jmap_annotations[i]] = element
    return this_map
