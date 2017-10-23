from itertools import izip_longest
import operator

orig_train = []
with open("train.txt") as f:
    for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
        #sentence = []
        orig_train.append(("<start>", "<startp>", "<starten>"))
        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
            orig_train.append((tok,p,i))

test = []
with open("test.txt") as f:
    for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
        orig_train.append(("<start>", "<startp>", "<starten>"))
        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
            test.append((tok,p,i))

train = orig_train
#train, test = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]
val = [("<start>", "<startp>", "<starten>")] + test

#print train
#prob(word1|entity)
def lexical(tup_arr):
    
    lex_list ={"<unk>": {}}
    
    for (tok, p, i) in tup_arr:
        if tok in lex_list:
            if i in lex_list[tok]:
                lex_list[tok][i] += 1
            else:
                lex_list[tok][i] = 1
        else:
            if i in lex_list["<unk>"]:
                lex_list["<unk>"][i] += 1
                lex_list[tok] = {i: 0}
            else:
                lex_list["<unk>"][i] = 1
                lex_list[tok] = {i: 0}
                
    return lex_list

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


def unigramCount(tup_arr):
    unigram_list = {"B-PER":0, "I-PER":0, "B-LOC":0, "I-LOC":0, "<starten>" : 0,
                   "B-ORG": 0, "I-ORG":0, "B-MISC":0, "I-MISC":0, "O": 0}
    
    for (tok, p, i) in tup_arr:
        if i in unigram_list:
            unigram_list[i] += 1
        else:
            print i + "is not a valid iob"

    return unigram_list

def unigram_token(tup_arr):
    unigrams = {"<unk>": 0}
    for (tok, p, i) in tup_arr:
			if tok in unigrams:
				unigrams[tok] += 1
			else:
				unigrams["<unk>"] +=1
				unigrams[tok] = 0

    return unigrams

def getlexicalProb(tup_arr, lex_list, token_list):
    voc = len(lex_list)
    return {k1: {k2: float (v2+1) / float (token_list[k1]+ 1*voc) for k2,v2 in v1.items()} 
              for k1,v1 in lex_list.items()}
    #{word: {entity1: prob(w|entity), entity2: prob(w|entity2)}}

    
def getTransitionProb(unigrams, transition):
    voc = len(unigrams)

    return {k1: {k2: float (v2+1) / float (unigrams[k1]+ 1*voc) for k2,v2 in v1.items()} 
              for k1,v1 in transition.items()}
    #{entitiy1: {entity2: prob}}
    
######actuall viterbi algoritm###############
#print getlexicalProb(train, lexical(train), unigram_token(train))
   
possible_entities = ["B-PER", "I-PER", "B-LOC","I-LOC", "<starten>","B-ORG",
                        "I-ORG", "B-MISC", "I-MISC", "O"]
score = {}
backpointer = {}
ans_dict = {}

token_list = unigram_token(train)
lex_list = lexical(train)
unigrams = unigramCount(train)
trans = transition(train) 
lexical_prob = getlexicalProb(train, lex_list, token_list)
transition_prob = getTransitionProb(unigrams, trans)

##for initialization
for (tok1, p1, iob1), (tok2, p2, iob2) in zip(val, val[1:]):
    
    if iob1 == "<starten>":
        for entity in possible_entities:
            if (tok2 in lexical_prob) and (entity in transition_prob):
                if entity in lexical_prob[tok2]:
                    lexicalProb = lexical_prob[tok2][entity]
                    if "<starten>" in transition_prob[entity]:
                        transitionProb = transition_prob[entity]["<starten>"]
                    else:
                        transitionProb = 1/float (unigrams[entity]+ len(unigrams))
                else:
                    lexicalProb = 1/float(token_list[tok2] + len(lex_list))
                    transitionProb = 1/float (unigrams[entity]+ len(unigrams))
            else: #new/unseen word
                lexicalProb = lexical_prob["<unk>"][entity]
                try:
                    transitionProb = transition_prob[entity]["<starten>"]
                except KeyError:
                    transitionProb = 0

            if tok2 in score:
                score[tok2][entity] =  lexicalProb*transitionProb
            else:
                score[tok2] = {entity: lexicalProb*transitionProb}
            if tok2 in backpointer:
                backpointer[tok2][entity] = {entity: "<starten>"}
            else:
                backpointer[tok2] = {entity: "<starten>"}
            ans_dict["<start>"] = "<starten>"
            
##iteration
    else:
    
        max_score_ent = max(score[tok1].iteritems(), key = operator.itemgetter(1))[0]
        print max_score
        max_score = score[tok1][max_score_ent]
        print max_score
        for entity in possible_entities:
            if (tok2 in lexical_prob) and (entity in transition_prob):
                if entity in lexical_prob[tok2]:
                    lexicalProb = lexical_prob[tok2][entity]
                    if max_score_ent in transition_prob[entity]:
                        transitionProb = transition_prob[entity][max_score_ent]
                    else:
                        transitionProb = 1/float (unigrams[entity]+ len(unigrams))   
                else:
                    lexicalProb = 1/float(token_list[tok2] + len(lex_list))
                    transitionProb = 1/float (unigrams[entity]+ len(unigrams))
                
                if tok2 in score:
                    score[tok2][entity] = max_score*lexicalProb*transitionProb
                else:
                    score[tok2] = {entity: max_score*lexicalProb*transitionProb}
                
                if tok2 in backpointer:
                    backpointer[tok2][entity] =  max_score_ent
                else:
                    backpointer[tok2] = {entity: max_score_ent}
            else:
                if entity in lexical_prob["<unk>"]:
                    lexicalProb = lexical_prob["<unk>"][entity]
                    if max_score_ent in transition_prob[entity]:
                        transitionProb = transition_prob[entity][max_score_ent]
                    else:
                        transitionProb = 1/ float (unigrams[entity]+ len(unigrams))
                else:
                   lexicalProb = 1/float(token_list[tok2] + len(lex_list))
                   transitionProb = 1/float (unigrams[entity]+ len(unigrams))
                
                if tok2 in score:
                    score[tok2][entity] = max_score*lexicalProb*transitionProb
                else:
                    score[tok2] = {entity: max_score*lexicalProb*transitionProb}
                
                if tok2 in backpointer:
                    backpointer[tok2][entity] =  max_score_ent
                else:
                    backpointer[tok2] = {entity: max_score_ent}
                
        ans_dict[tok2] = backpointer[tok2][max_score_ent]
print ans_dict

def test_entity_index(iobs, indicies):
    org, misc, per, loc = [], [], [], []
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

# get test data entity predictions
pred_toks = [ans_dict[tok] for tok, pos, ind in test]
pred_ind = [ind for tok, pos, ind in test]

org_pred, misc_pred, per_pred, loc_pred = test_entity_index(pred_toks, pred_ind)

# output the results in file named output.txt
output = open("output.txt", "w")
output.write("Type,Prediction\n")
output.write("ORG," + " ".join(org_pred) + "\n")
output.write("MISC," + " ".join(misc_pred) + "\n")
output.write("PER," + " ".join(per_pred) + "\n")
output.write("LOC," + " ".join(loc_pred))

