from itertools import zip_longest
from collections import defaultdict

O = 0
B_ORG = 1
I_ORG = 2
B_MISC = 3
I_MISC = 4
B_PER = 5
I_PER = 6
B_LOC = 7
I_LOC = 8
words_iob_counts = defaultdict(list)

"""
def check(words, pos, i):
	return words[i][0].isupper() and pos[i] == "NNP"
def trivial():
    all_ranges = []
    with open("test.txt") as f:
        while True:
            words, pos, index = [f.readline().rstrip().split() for _ in range(3)]
            if not words or not pos or not index:
                break
            i = 0
            while i < len(words):
                if check(words, pos, i):
                    range_index = index[i]
                    while i+1 < len(words) and check(words, pos, i+1):
                        i += 1
                    all_ranges.append(str(range_index) + "-" + str(index[i]))
                i += 1

    output = open("output.txt", "w")
    output.write("Type,Prediction\n")
    output.write("ORG," + " ".join(all_ranges[:len(all_ranges)/4]) + "\n")
    output.write("MISC," + " ".join(all_ranges[len(all_ranges)/4:len(all_ranges)*2/4]) + "\n")
    output.write("PER," + " ".join(all_ranges[len(all_ranges)*2/4:len(all_ranges)*3/4]) + "\n")
    output.write("LOC," + " ".join(all_ranges[len(all_ranges)*3/4:]))
"""

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
    words = []
    pos_tags = []
    indicies = []
    with open(text) as f:
        for toks, pos, ind in zip_longest(*[f]*3):
            sentence = []
            line_pos = []
            line_ind = []
            for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), ind.rstrip().split()):
                sentence.append(tok)
                line_pos.append(p)
                line_ind.append(i)
            words.append(sentence)
            pos_tags.append(line_pos)
            indicies.append(line_ind)
    return words, pos_tags, indicies

def get_iobs(line):
    return [iob for tok, pos, iob in line]

def get_tokens(line):
    return [tok for tok, pos, iob in line]

def get_pos(line):
    return [pos for tok, pos, iob in line]


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


def iob_counts(train):
    for line in train:
        for tok, pos, iob in line:
            count = [0] * 9
            if tok in words_iob_counts:
                count = words_iob_counts[tok]
            if iob == 'O':
                count[O] += 1
            elif iob == 'B-ORG':
                count[B_ORG] += 1
            elif iob == 'I-ORG':
                count[I_ORG] += 1
            elif iob == 'B-MISC':
                count[B_MISC] += 1
            elif iob == 'I-MISC':
                count[I_MISC] += 1
            elif iob == 'B-PER':
                count[B_PER] += 1
            elif iob == 'I-PER':
                count[I_PER] += 1
            elif iob == 'B-LOC':
                count[B_LOC] += 1
            elif iob == 'I-LOC':
                count[I_LOC] += 1
            words_iob_counts[tok] = count

def set_tag(test, pos_tags):
    iob_preds = []
    for line, pos_line in zip(test, pos_tags):
        line_pred = []
        for tok, pos in zip(line, pos_line):
            if tok in words_iob_counts:
                ind = words_iob_counts[tok].index(max(words_iob_counts[tok]))
                if ind == O:
                    line_pred.append('O')
                elif ind == B_ORG:
                    line_pred.append('B-ORG')
                elif ind == I_ORG:
                    line_pred.append('I-ORG')
                elif ind == B_MISC:
                    line_pred.append('B-MISC')
                elif ind == I_MISC:
                    line_pred.append('I-MISC')
                elif ind == B_PER:
                    line_pred.append('B-PER')
                elif ind == I_PER:
                    line_pred.append('I-PER')
                elif ind == B_LOC:
                    line_pred.append('B-LOC')
                elif ind == I_LOC:
                    line_pred.append('I-LOC')
            else:
                line_pred.append('O')
        iob_preds.append(line_pred)
    return iob_preds

def main():
    # getting the training data
    orig_train = get_train("train.txt")
    
    # training data is split so that first 90% is the training set and the last 10% is the validation set
    train, valid = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]
    toks_valid = [get_tokens(line) for line in valid]
    pos_valid = [get_pos(line) for line in valid]
    iobs_valid = [get_iobs(line) for line in valid]
    org_true, misc_true, per_true, loc_true = entity_index(iobs_valid)

    """
        Baseline
        Percision: 0.634167385677308
        Recall: 0.5944197331176708
        F1-score: 0.6136505948653725
        Kaggle: 0.64833
    """
    # training and applying baseline on validation set
    iob_counts(train)
    pred_valid = set_tag(toks_valid, pos_valid)

    # accuracy measures
    calculate_measures(org_true, misc_true, per_true, loc_true, pred_valid, "Baseline")


    # get test data, apply baseline, and get entity predictions
    toks, pos, indicies = get_test("test.txt")
    pred_test = set_tag(toks, pos)
    org_pred, misc_pred, per_pred, loc_pred = test_entity_index(pred_test, indicies)
    
    # output the results in file named output.txt
    output = open("output.txt", "w")
    output.write("Type,Prediction\n")
    output.write("ORG," + " ".join(org_pred) + "\n")
    output.write("MISC," + " ".join(misc_pred) + "\n")
    output.write("PER," + " ".join(per_pred) + "\n")
    output.write("LOC," + " ".join(loc_pred))

    
if __name__ == '__main__':
    main()

