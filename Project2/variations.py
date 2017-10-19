# NOTE: MUST USE PYTHON3 
from itertools import chain, zip_longest

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics

# Text file is represented by orig_train, a list of lists. The inner lists represents a sentence or a line in the file.
# The sentence is actually a list of lists as well, where each inner list contains a word token, the pos tag of the token, and the iob tag of the token.
def preprocessing(text):
    orig_train = []
    with open(text) as f:
        for toks, pos, iob in zip_longest(*[f]*3):
            sentence = []
            for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
                sentence.append((tok,p,i))
            orig_train.append(sentence)
    return orig_train

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


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
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
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
                        '-1:word.lower()': word1.lower(),
                        '-1:word.istitle()': word1.istitle(),
                        '-1:word.isupper()': word1.isupper(),
                        '-1:postag': postag1,
                        '-1:postag[:2]': postag1[:2],
                        })
    else:
        features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
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


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def main():
    # getting the training data
    orig_train = preprocessing("train.txt")

    # training data is split so that first 90% is the training set and the last 10% is the validation set
    train, test = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]

    # extracting features for word tokens
    X_train = [sent2features(s) for s in train]
    y_train = [sent2labels(s) for s in train]
    X_test = [sent2features(s) for s in test]
    y_test = [sent2labels(s) for s in test]

    # training data using CRF with L-BFGS training algorithm and Elastic Net (L1 + L2) regularization
    crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs', c1 = 0.1, c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test) # testing with validation set

    # Accuracy Measures
    org_pred, misc_pred, per_pred, loc_pred = entity_index(y_pred)
    org_true, misc_true, per_true, loc_true = entity_index(y_test)
    print("Type\tPercision\t\tRecall\t\t\tF-score")
    print("ORG\t" + "\t".join(str(x) for x in measures(org_true, org_pred)))
    print("MISC\t" + "\t".join(str(x) for x in measures(misc_true, misc_pred)))
    print("PER\t" + "\t".join(str(x) for x in measures(per_true, per_pred)))
    print("LOC\t" + "\t".join(str(x) for x in measures(loc_true, loc_pred)))


if __name__ == '__main__':
    main()





