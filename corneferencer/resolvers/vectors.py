from corneferencer.resolvers import features


def get_pair_vector(ante, ana):
    vec = []
    vec.extend(ante.features)
    vec.extend(ana.features)
    pair_features = get_pair_features(ante, ana)
    vec.extend(pair_features)
    # print(f"vectors.get_pair_vector {len(ante.features)} {len(ana.features)} {len(pair_features)} {len(vec)}")
    return vec


def get_mention_features(mention):
    vec = []
    vec.extend(features.head_vec(mention))
    vec.extend(features.first_word_vec(mention))
    vec.extend(features.last_word_vec(mention))
    vec.extend(features.first_after_vec(mention))
    vec.extend(features.second_after_vec(mention))
    vec.extend(features.first_before_vec(mention))
    vec.extend(features.second_before_vec(mention))
    vec.extend(features.preceding_context_vec(mention))
    vec.extend(features.following_context_vec(mention))
    vec.extend(features.mention_vec(mention))
    vec.extend(features.sentence_vec(mention))

    # complementary features
    vec.extend(features.mention_type(mention))

    # complementary features 2
    vec.append(features.is_first_second_person(mention))
    vec.append(features.is_demonstrative(mention))
    vec.append(features.is_demonstrative_nominal(mention))
    vec.append(features.is_demonstrative_pronoun(mention))
    vec.append(features.is_refl_pronoun(mention))
    vec.append(features.is_first_in_sentence(mention))
    vec.append(features.is_zero_or_pronoun(mention))
    vec.append(features.head_contains_digit(mention))
    vec.append(features.mention_contains_digit(mention))
    vec.append(features.contains_letter(mention))
    vec.append(features.post_modified(mention))

    return vec


def get_pair_features(ante, ana):
    vec = []
    vec.extend(features.distances_vec(ante, ana))
    vec.append(features.head_match(ante, ana))
    vec.append(features.exact_match(ante, ana))
    vec.append(features.base_match(ante, ana))

    # complementary features
    vec.append(features.ante_contains_rarest_from_ana(ante, ana))
    vec.extend(features.agreement(ante, ana, 'gender'))
    vec.extend(features.agreement(ante, ana, 'number'))
    vec.extend(features.agreement(ante, ana, 'person'))
    vec.append(features.is_acronym(ante, ana))
    vec.append(features.same_sentence(ante, ana))
    vec.append(features.same_paragraph(ante, ana))

    # complementary features 2
    vec.append(features.neighbouring_sentence(ante, ana))
    vec.append(features.cousin_sentence(ante, ana))
    vec.append(features.distant_sentence(ante, ana))
    vec.extend(features.flat_gender_agreement(ante, ana))
    vec.append(features.left_match(ante, ana))
    vec.append(features.right_match(ante, ana))
    vec.append(features.abbrev2(ante, ana))

    vec.append(features.string_kernel(ante, ana))
    vec.append(features.head_string_kernel(ante, ana))

    vec.append(features.wordnet_synonyms(ante, ana))
    vec.append(features.wordnet_ana_is_hypernym(ante, ana))
    vec.append(features.wordnet_ante_is_hypernym(ante, ana))

    vec.append(features.wikipedia_link(ante, ana))
    vec.append(features.wikipedia_mutual_link(ante, ana))
    vec.append(features.wikipedia_redirect(ante, ana))

    # combined features
    vec.append(features.samesent_anapron_antefirstinpar(ante, ana))
    vec.append(features.samesent_antefirstinpar_personnumbermatch(ante, ana))
    vec.append(features.adjsent_anapron_adjmen_personnumbermatch(ante, ana))
    vec.append(features.adjsent_anapron_adjmen(ante, ana))

    return vec
