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


# calculates the precision, recall, and f1-score at entity level and prints out the results
def calculate_measures(org_gold, misc_gold, per_gold, loc_gold, iob_predict, model_describe):
    org_pred, misc_pred, per_pred, loc_pred = entity_index(iob_predict)
    
    # accuracy measures
    true_positive = float(sum([1 for org in org_gold if org in org_pred])
                    + sum([1 for misc in misc_gold if misc in misc_pred])
                    + sum([1 for per in per_gold if per in per_pred])
                    + sum([1 for loc in loc_gold if loc in loc_pred]))

    gold = len(org_gold) + len(misc_gold) + len(per_gold) + len(loc_gold)
    pred = len(org_pred) + len(misc_pred) + len(per_pred) + len(loc_pred)

    print("\n" + model_describe)
    print("Percision: " + str(true_positive/pred))
    print("Recall: " + str(true_positive/gold))
    print("F1-score: " + str(2*true_positive/(pred+gold)))


def word_to_features(line, ind):
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
    return [word_to_features(line, ind) for ind in range(len(line))]

def get_iobs(line):
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
    y_train = [get_iobs(line) for line in train]
    X_valid = [get_features(line) for line in valid]
    y_valid = [get_iobs(line) for line in valid]
    org_true, misc_true, per_true, loc_true = entity_index(y_valid)


    # training data using CRF with L-BFGS training algorithm and Elastic Net (L1 + L2) regularization
    lbfgs_crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs', c1 = 0.1, c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    lbfgs_crf.fit(X_train, y_train)
    iobs = list(lbfgs_crf.classes_)
    y_pred = lbfgs_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Gradient descent using the L-BFGS method")
    
    # feature statistics
    print_features_stats(lbfgs_crf)


    # training data using CRF with Stochastic Gradient Descent training algorithm and Elastic Net (L2) regularization
    l2sgd_crf = sklearn_crfsuite.CRF(algorithm = 'l2sgd', c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    l2sgd_crf.fit(X_train, y_train)
    iobs = list(l2sgd_crf.classes_)
    y_pred = l2sgd_crf.predict(X_valid) # testing with validation set

    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Stochastic Gradient Descent with L2 regularization term")

    # feature statistics
    print_features_stats(l2sgd_crf)

    # training data using CRF with Averaged Perceptron training algorithm
    ap_crf = sklearn_crfsuite.CRF(algorithm = 'ap', max_iterations = 100, all_possible_transitions = True)
    ap_crf.fit(X_train, y_train)
    iobs = list(ap_crf.classes_)
    y_pred = ap_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Averaged Perceptron")
    
    # feature statistics
    print_features_stats(ap_crf)
    

    # training data using CRF with Passive Aggressive training algorithm
    pa_crf = sklearn_crfsuite.CRF(algorithm = 'pa', max_iterations = 100, all_possible_transitions = True)
    pa_crf.fit(X_train, y_train)
    iobs = list(pa_crf.classes_)
    y_pred = pa_crf.predict(X_valid) # testing with validation set
    
    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, y_pred, "CRF: Passive Aggressive")
    
    # feature statistics
    print_features_stats(pa_crf)


    # training data using CRF with Adaptive Regularization Of Weight Vector training algorithm
    arow_crf = sklearn_crfsuite.CRF(algorithm = 'arow', max_iterations = 100, all_possible_transitions = True)
    arow_crf.fit(X_train, y_train)
    iobs = list(arow_crf.classes_)
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
    y_pred = arow_crf.predict(X_test)
    org_pred, misc_pred, per_pred, loc_pred = test_entity_index(y_pred, indicies)

    # output the results in file named output.txt
    output = open("output.txt", "w")
    output.write("Type,Prediction\n")
    output.write("ORG," + " ".join(org_pred) + "\n")
    output.write("MISC," + " ".join(misc_pred) + "\n")
    output.write("PER," + " ".join(per_pred) + "\n")
    output.write("LOC," + " ".join(loc_pred))


if __name__ == '__main__':
    main()





