import numpy

from corneferencer.resolvers import features, vectors


def siamese(text, threshold, neural_model):
    last_set_id = 0
    for i, ana in enumerate(text.mentions):
        if i > 0:
            for ante in reversed(text.mentions[:i]):
                if not features.pair_intersect(ante, ana):
                    pair_features = vectors.get_pair_features(ante, ana)

                    ante_vec = []
                    ante_vec.extend(ante.features)
                    ante_vec.extend(pair_features)
                    ante_sample = numpy.asarray([ante_vec], dtype=numpy.float32)

                    ana_vec = []
                    ana_vec.extend(ana.features)
                    ana_vec.extend(pair_features)
                    ana_sample = numpy.asarray([ana_vec], dtype=numpy.float32)

                    prediction = neural_model.predict([ante_sample, ana_sample])[0]

                    if prediction < threshold:
                        if ante.set:
                            ana.set = ante.set
                        else:
                            str_set_id = 'set_%d' % last_set_id
                            ante.set = str_set_id
                            ana.set = str_set_id
                            last_set_id += 1
                        break


# incremental resolve algorithm
def incremental(text, threshold, neural_model):
    last_set_id = 0
    for i, ana in enumerate(text.mentions):
        if i > 0:
            best_prediction = 0.0
            best_ante = None
            for ante in text.mentions[:i]:
                if not features.pair_intersect(ante, ana):
                    pair_vec = vectors.get_pair_vector(ante, ana)
                    sample = numpy.asarray([pair_vec], dtype=numpy.float32)
                    prediction = neural_model.predict(sample)[0]
                    if prediction > threshold and prediction >= best_prediction:
                        best_prediction = prediction
                        best_ante = ante
            if best_ante is not None:
                if best_ante.set:
                    ana.set = best_ante.set
                else:
                    str_set_id = 'set_%d' % last_set_id
                    best_ante.set = str_set_id
                    ana.set = str_set_id
                    last_set_id += 1


# all2all resolve algorithm
def all2all(text, threshold, neural_model):
    last_set_id = 0
    sets = text.get_sets()
    for pos1, mnt1 in enumerate(text.mentions):
        best_prediction = 0.0
        best_link = None
        for pos2, mnt2 in enumerate(text.mentions):
            if (pos2 > pos1 and
                    (mnt1.set != mnt2.set or not mnt1.set or not mnt2.set)
                    and not features.pair_intersect(mnt1, mnt2)):
                ante = mnt1
                ana = mnt2
                pair_vec = vectors.get_pair_vector(ante, ana)
                sample = numpy.asarray([pair_vec], dtype=numpy.float32)
                prediction = neural_model.predict(sample)[0]
                if prediction > threshold and prediction > best_prediction:
                    best_prediction = prediction
                    best_link = mnt2
        if best_link is not None:
            if best_link.set and not mnt1.set:
                mnt1.set = best_link.set
            elif not best_link.set and mnt1.set:
                best_link.set = mnt1.set
            elif best_link.set and mnt1.set:
                text.merge_sets(best_link.set, mnt1.set)
            elif not best_link.set and not mnt1.set:
                str_set_id = 'set_%d' % last_set_id
                while str_set_id in sets:
                    last_set_id += 1
                    str_set_id = 'set_%d' % last_set_id
                best_link.set = str_set_id
                mnt1.set = str_set_id
                sets[str_set_id] = [best_link, mnt1]


# entity based resolve algorithm
def entity_based(text, threshold, neural_model):
    sets = []
    last_set_id = 0
    for i, ana in enumerate(text.mentions):
        if i > 0:
            best_fit = get_best_set(sets, ana, threshold, neural_model)
            if best_fit is not None:
                ana.set = best_fit['set_id']
                best_fit['mentions'].append(ana)
            else:
                str_set_id = 'set_%d' % last_set_id
                sets.append({'set_id': str_set_id,
                             'mentions': [ana]})
                ana.set = str_set_id
                last_set_id += 1
        else:
            str_set_id = 'set_%d' % last_set_id
            sets.append({'set_id': str_set_id,
                         'mentions': [ana]})
            ana.set = str_set_id
            last_set_id += 1

    remove_singletons(sets)


def get_best_set(sets, ana, threshold, neural_model):
    best_prediction = 0.0
    best_set = None
    for s in sets:
        accuracy = predict_set(s['mentions'], ana, neural_model)
        if accuracy > threshold and accuracy >= best_prediction:
            best_prediction = accuracy
            best_set = s
    return best_set


def predict_set(mentions, ana, neural_model):
    prediction_sum = 0.0
    for mnt in mentions:
        prediction = 0.0
        if not features.pair_intersect(mnt, ana):
            pair_vec = vectors.get_pair_vector(mnt, ana)
            sample = numpy.asarray([pair_vec], dtype=numpy.float32)
            prediction = neural_model.predict(sample)[0]
        prediction_sum += prediction
    return prediction_sum / float(len(mentions))


def remove_singletons(sets):
    for s in sets:
        if len(s['mentions']) == 1:
            s['mentions'][0].set = ''


# closest resolve algorithm
def closest(text, threshold, neural_model):
    last_set_id = 0
    for i, ana in enumerate(text.mentions):
        if i > 0:
            for ante in reversed(text.mentions[:i]):
                if not features.pair_intersect(ante, ana):
                    pair_vec = vectors.get_pair_vector(ante, ana)
                    sample = numpy.asarray([pair_vec], dtype=numpy.float32)
                    prediction = neural_model.predict(sample)[0]
                    if prediction > threshold:
                        if ante.set:
                            ana.set = ante.set
                        else:
                            str_set_id = 'set_%d' % last_set_id
                            ante.set = str_set_id
                            ana.set = str_set_id
                            last_set_id += 1
                        break
