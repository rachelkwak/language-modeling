# NOTE: MUST USE PYTHON3+

from itertools import chain, zip_longest
from collections import Counter

import nltk
import sklearn
import sklearn_crfsuite

# Text file is represented by orig_train, a list of lists. The inner lists represents a sentence or a line in the file.
# The sentence is actually a list of lists as well, where each inner list contains a word token, the pos tag of the token, and the iob tag of the token.
def get_train(text):
    orig_train = []
    with open(text) as f:
        for toks, pos, iob in zip_longest(*[f]*3):
            sentence = []
            for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
                sentence.append((tok,p,i))
            orig_train.append(sentence)
    return orig_train

# Returns a list similar to get_train except there is no iob tag component
# Also returns a list of list of indicies that match the position of each word token
def get_test(text):
    test = []
    indicies = []
    with open(text) as f:
        for toks, pos, ind in zip_longest(*[f]*3):
            sentence = []
            line_pos = []
            for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), ind.rstrip().split()):
                sentence.append((tok,p))
                line_pos.append(i)
            test.append(sentence)
            indicies.append(line_pos)
    return test, indicies

# Takes in iob tags of word tokens as a list of lists (each inner list is a sentence)
# The index of the word token in the corpus should correspond with the index of its iob tag in the flatten list of iobs
# Returns the list of entities for each type: ORG, MISC, PER, and LOC; in this order
def entity_index(iobs):
    org, misc, per, loc = [], [], [], []
    iobs = [item for sublist in iobs for item in sublist]
    ind = 0
    while ind < len(iobs):
        if iobs[ind] != "O":
            type = iobs[ind]
            range_ind = ind
            while ind+1 < len(iobs) and iobs[ind+1] != type and type[type.index('-')+1:] in iobs[ind+1]:
                ind += 1
            range = str(range_ind) + "-" + str(ind)
            if "ORG" in type:
                org.append(range)
            elif "MISC" in type:
                misc.append(range)
            elif "PER" in type:
                per.append(range)
            else:
                loc.append(range)
        ind += 1
    return org, misc, per, loc

# Similar to entity_index except the positions for word token are provided and used in listing of entities
def test_entity_index(iobs, indicies):
    org, misc, per, loc = [], [], [], []
    iobs = [item for sublist in iobs for item in sublist]
    indicies = [item for sublist in indicies for item in sublist]
    ind = 0
    while ind < len(iobs):
        if iobs[ind] != "O":
            type = iobs[ind]
            range_ind = indicies[ind]
            while ind+1 < len(iobs) and iobs[ind+1] != type and type[type.index('-')+1:] in iobs[ind+1]:
                ind += 1
            range = str(range_ind) + "-" + str(indicies[ind])
            if "ORG" in type:
                org.append(range)
            elif "MISC" in type:
                misc.append(range)
            elif "PER" in type:
                per.append(range)
            else:
                loc.append(range)
        ind += 1
    return org, misc, per, loc


# Takes in list of correct entities and list of predicted entities
# Returns precision, recall, and f-score measures in this order
def measures(actual, predict):
    true_pos = 0
    for range in actual:
        if range in predict:
            true_pos += 1
    if true_pos == 0:
        return 0, 0, 0
    p = true_pos / len(predict)
    r = true_pos / len(actual)
    f = (2 * p * r) / (p + r)
    return p, r, f

# calculates the precision, recall, and f-score for all entities and prints out the results
def calculate_measures(org_true, misc_true, per_true, loc_true, iob_predict, model_describe):
    org_pred, misc_pred, per_pred, loc_pred = entity_index(iob_predict)
    
    # accuracy measures
    org_measure = measures(org_true, org_pred)
    misc_measure = measures(misc_true, misc_pred)
    per_measure = measures(per_true, per_pred)
    loc_measure = measures(loc_true, loc_pred)
    
    # print results
    print("\n" + model_describe)
    print("Type\tPercision\t\tRecall\t\t\tF-score")
    print("ORG\t" + "\t".join(str(x) for x in org_measure))
    print("MISC\t" + "\t".join(str(x) for x in misc_measure))
    print("PER\t" + "\t".join(str(x) for x in per_measure))
    print("LOC\t" + "\t".join(str(x) for x in loc_measure))
    print("Averge\t" + str((org_measure[0]+misc_measure[0]+per_measure[0]+loc_measure[0])/4)
          + "\t" + str((org_measure[1]+misc_measure[1]+per_measure[1]+loc_measure[1])/4)
          + "\t" + str((org_measure[2]+misc_measure[2]+per_measure[1]+loc_measure[2])/4))


def word2features(line, ind):
    word = line[ind][0]
    postag = line[ind][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if ind > 0:
        word1 = line[ind-1][0]
        postag1 = line[ind-1][1]
        features.update({
                        '-1:word.lower()': word1.lower(),
                        '-1:word.istitle()': word1.istitle(),
                        '-1:word.isupper()': word1.isupper(),
                        '-1:postag': postag1,
                        '-1:postag[:2]': postag1[:2],
                        })
    else:
        features['BOS'] = True
    if ind < len(line)-1:
        word1 = line[ind+1][0]
        postag1 = line[ind+1][1]
        features.update({
                        '+1:word.lower()': word1.lower(),
                        '+1:word.istitle()': word1.istitle(),
                        '+1:word.isupper()': word1.isupper(),
                        '+1:postag': postag1,
                        '+1:postag[:2]': postag1[:2],
                        })
    else:
        features['EOS'] = True
    return features

def get_features(line):
    return [word2features(line, ind) for ind in range(len(line))]

def get_labels(line):
    return [iob for tok, pos, iob in line]

def get_tokens(line):
    return [tok for tok, pos, iob in line]

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

def print_features_stats(crf):
    print("\nTop likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))
    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
    print("\nTop positive:")
    print_state_features(Counter(crf.state_features_).most_common(20))
    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-20:])

def main():
    # getting the training data
    orig_train = get_train("train.txt")

    # training data is split so that first 90% is the training set and the last 10% is the validation set
    train, valid = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]
    
    # extracting features for word tokens
    X_train = [get_features(line) for line in train]
    y_train = [get_labels(line) for line in train]
    X_valid = [get_features(line) for line in valid]
    y_valid = [get_labels(line) for line in valid]
    org_true, misc_true, per_true, loc_true = entity_index(y_valid)


    # training data using CRF with L-BFGS training algorithm and Elastic Net (L1 + L2) regularization
    lbfgs_crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs', c1 = 0.1, c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    lbfgs_crf.fit(X_train, y_train)
    labels = list(lbfgs_crf.classes_)
    y_pred = lbfgs_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Gradient descent using the L-BFGS method")
    
    # feature statistics
    print_features_stats(lbfgs_crf)
    

    # training data using CRF with Stochastic Gradient Descent training algorithm and Elastic Net (L2) regularization
    l2sgd_crf = sklearn_crfsuite.CRF(algorithm = 'l2sgd', c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    l2sgd_crf.fit(X_train, y_train)
    labels = list(l2sgd_crf.classes_)
    y_pred = l2sgd_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Stochastic Gradient Descent with L2 regularization term")

    # feature statistics
    print_features_stats(l2sgd_crf)

    # training data using CRF with Averaged Perceptron training algorithm
    ap_crf = sklearn_crfsuite.CRF(algorithm = 'ap', max_iterations = 100, all_possible_transitions = True)
    ap_crf.fit(X_train, y_train)
    labels = list(ap_crf.classes_)
    y_pred = ap_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Averaged Perceptron")
    
    # feature statistics
    print_features_stats(ap_crf)
    

    # training data using CRF with Passive Aggressive training algorithm
    pa_crf = sklearn_crfsuite.CRF(algorithm = 'pa', max_iterations = 100, all_possible_transitions = True)
    pa_crf.fit(X_train, y_train)
    labels = list(pa_crf.classes_)
    y_pred = pa_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Averaged Perceptron")
    
    # feature statistics
    print_features_stats(pa_crf)


    # training data using CRF with Adaptive Regularization Of Weight Vector training algorithm
    arow_crf = sklearn_crfsuite.CRF(algorithm = 'arow', max_iterations = 100, all_possible_transitions = True)
    arow_crf.fit(X_train, y_train)
    labels = list(arow_crf.classes_)
    y_pred = arow_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Adaptive Regularization Of Weight Vector")

    # feature statistics
    print_features_stats(arow_crf)


    # using test data on best model
    # get test data
    test, indicies = get_test("test.txt")

    # get predictions for entities
    X_test = [get_features(line) for line in test]
    y_pred = pa_crf.predict(X_test)
    org_pred, misc_pred, per_pred, loc_pred = test_entity_index(y_pred, indicies)

    # output the results in file named output.txt
    output = open("output.txt", "w")
    output.write("Type,Prediction\n")
    output.write("ORG," + " ".join(org_pred) + "\n")
    output.write("MISC," + " ".join(misc_pred) + "\n")
    output.write("PER," + " ".join(per_pred) + "\n")
    output.write("LOC," + " ".join(loc_prred))


if __name__ == '__main__':
    main()





