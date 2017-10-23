from itertools import izip_longest
import math
import operator

possible_entities = ["B-PER", "I-PER", "B-LOC","I-LOC", "<starten>","B-ORG",
                        "I-ORG", "B-MISC", "I-MISC", "O"]

orig_train = []
with open("train.txt") as f:
    for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
        orig_train.append(("<start>", "<startp>", "<starten>"))
        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
            orig_train.append((tok,p,i))

#print train
#prob(word1|entity)
def train_lexical(tup_arr):
    
    lex_list ={"<unk>": {}}
    
    for (tok, p, i) in tup_arr:
        if tok in lex_list:
            if i in lex_list[tok]:
                lex_list[tok][i] += 1
            else:
                lex_list[tok][i] = 1
        else:
            lex_list[tok] = {i:1}
                
    return lex_list
            

train, test = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]
val = [("<start>", "<startp>", "<starten>")] + test

train_lexical = train_lexical(orig_train)


test_file = []
with open("test.txt") as f:
    for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
        orig_train.append(("<start>", "<startp>", "<starten>"))
        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
            if tok in train_lexical:
                test_file.append((tok,p,i))
            else:
                test_file.append(("<unk>",p,i))




def transition(tup_arr):
    transition_list = {"B-PER":{}, "I-PER":{}, "B-LOC":{}, "I-LOC":{}, "<starten>": {},
                   "B-ORG": {}, "I-ORG":{}, "B-MISC":{}, "I-MISC":{}, "O": {}}

    for (tok1, p1, i1), (tok2, p2, i2) in zip(tup_arr, tup_arr[1:]):
		if tok1 != ".":
			if i1 in transition_list:
				if i2 in transition_list[i1]:
					transition_list[i1][i2] += 1
				else:
					transition_list[i1][i2] = 1
			else:
				print "invalid bigram iod"
    return transition_list

train_transitions = transition(train)

def unigramCount(tup_arr):
    unigram_list = {"B-PER":0, "I-PER":0, "B-LOC":0, "I-LOC":0, "<starten>" : 0,
                   "B-ORG": 0, "I-ORG":0, "B-MISC":0, "I-MISC":0, "O": 0}
    
    for (tok, p, i) in tup_arr:
        if i in unigram_list:
            unigram_list[i] += 1
        else:
            print i + "is not a valid iob"

    return unigram_list

unigram_counts = unigramCount(train)

def unigram_token(tup_arr):
    unigrams = {"<unk>": 0}
    for (tok, p, i) in tup_arr:
		if tok in unigrams:
			unigrams[tok] += 1
		else:
			unigrams["<unk>"] +=1
			unigrams[tok] = 0

    return unigrams

def getlexicalProb(lex_list, unigram_counts):
    """
    voc = len(lex_list)
    return {k1: {k2: float (v2+1) / float (token_list[k1]+ 1*voc) for k2,v2 in v1.items()} 
              for k1,v1 in lex_list.items()}
    #{word: {entity1: prob(w|entity), entity2: prob(w|entity2)}}
    """
    lex = lex_list.keys()
    lexical_table = [[0 for _ in xrange(len(possible_entities))] for _ in xrange(len(lex_list))]
    for i in range(len(lex)):
        for j in range(len(possible_entities)):
            a = lex[i]
            b = possible_entities[j]
            try:
                lexical_table[i][j] = math.log(float(lex_list[a][b])/unigram_counts[b])
            except KeyError:
                lexical_table[i][j] = math.log(1/(float(unigram_counts[b] + float(len(lex)))))
    return lexical_table

# print getlexicalProb(train_lexical, unigram_counts)

    
def getTransitionProb(unigram_counts, train_transitions):
    """
    voc = len(unigrams)

    return {k1: {k2: float (v2+1) / float (unigrams[k1]+ 1*voc) for k2,v2 in v1.items()} 
              for k1,v1 in transition.items()}
    #{entitiy1: {entity2: prob}}
    """
    transition_table = [[0 for _ in xrange(len(possible_entities))] for _ in xrange(len(possible_entities))]
    for i in range(len(possible_entities)):
        for j in range(len(possible_entities)):
            a = possible_entities[i]
            b = possible_entities[j]
            try:
                transition_table[i][j] = math.log(float(train_transitions[b][a]) / unigram_counts[b])
            except KeyError:
                transition_table[i][j] = math.log(1/(float(len(unigram_counts)) + float(unigram_counts[b])))
    for i in range(len(possible_entities)):
        transition_table[4][i] = -50
    return transition_table

# print getTransitionProb(unigram_counts, train_transitions)

lexical_prob = getlexicalProb(train_lexical, unigram_counts)
transition_prob = getTransitionProb(unigram_counts, train_transitions)

# w_t | t_i
# print lexical_prob['Bentsen']['I-PER']

# B-PER | O
# print transition_prob['B-PER']['O']
# print transition_prob


num_states = len(possible_entities)
num_obs = len(test_file) / 100

score = [[0 for i in xrange(num_obs)] for _ in xrange(num_states)]
bptr = [[0 for i in xrange(num_obs)] for _ in xrange(num_states)]

for i in xrange(num_states):
    score[i][0] = transition_prob[i][4] + lexical_prob[2912][i]


for t in xrange(1, num_obs):
    for i in xrange(num_states):
        max_score = -float('inf')
        max_index = 0

        for j in xrange(num_states):
            prev_max = max_score
            max_score = max(score[j][t-1] + transition_prob[i][j], max_score)
            
            if max_score != prev_max:
                max_index = j

        score[i][t] = max_score + lexical_prob[t][i]
        bptr[i][t] = max_index

T = [0 for _ in xrange(num_obs)]
max_T_index = 0
max_T = -float('inf')
for i in xrange(num_states):
    prev = max_T
    max_T = max(score[i][num_obs-1], max_T)
    if prev != max_T:
        max_T_index = i
T[num_obs-1] = max_T_index

for i in xrange(num_obs-2, -1, -1):
    T[i] = bptr[T[i+1]][i+1]

for i in xrange(num_obs):
    print test_file[i][0], "("+possible_entities[T[i]]+")"

