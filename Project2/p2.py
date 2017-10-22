from itertools import izip_longest
import operator

orig_train = []
with open("train.txt") as f:
    for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
        #sentence = []
        orig_train.append(("<start>", "<startp>", "<starten>"))
        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
            orig_train.append((tok,p,i))
        

train, test = orig_train[:int(len(orig_train)*.9)], orig_train[int(len(orig_train)*.9):]
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
					transition_list[i1] = {i2 : 1}
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
    return {k1: {k2: (v2+1) / (unigrams[k1]+ 1*voc) for k2,v2 in v1.items()} 
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


##for initialization
for (tok1, p1, iob1), (tok2, p2, iob2) in zip(test, test[1:]):
    if iob1 == "<starten>":
        for entity in possible_entities:
            if (tok2 in getlexicalProb(train, lex_list, token_list)) and (entity in getTransitionProb(unigrams, trans)):
                if entity in getlexicalProb(train, lex_list, token_list)[tok2]:
                    lexicalProb = getlexicalProb(train, lex_list, token_list)[tok2][entity]
                    if "<starten>" in getTransitionProb(unigrams, trans)[entity]:
                        transitionProb = getTransitionProb(unigrams, trans)[entity]["<starten>"]
                    else:
                        transitionProb = 1/(unigrams[entity]+ len(unigrams))
                else:
                    lexicalProb = 0
                    transitionProb = 0
            else: #new/unseen word
                lexicalProb = getlexicalProb(train, lex_list, token_list)["<unk>"][entity]
                transitionProb = getTransitionProb(unigrams, trans)[entity]["<start>"]
            
            score = {tok2:{entity : lexicalProb*transitionProb}}
            backpointer = {tok2: {entity: "<starten>"}}
            ans_dict = {"<start>": "<starten>"}
            
##iteration
    else:
        max_score_ent = max(score[tok1].iteritems(), key = operator.itemgetter(1))[0]
        max_score = score[tok1][max_score_ent]
        for entity in possible_entities:
            if (tok2 in getlexicalProb(train, lex_list, token_list)) and (entity in getTransitionProb(unigrams, trans)):
                if entity in getlexicalProb(train, lex_list, token_list)[tok2]:
                    lexicalProb = getlexicalProb(train, lex_list, token_list)[tok2][entity]
                    if max_score_ent in getTransitionProb(unigrams, trans)[entity]:
                        transitionProb = getTransitionProb(unigrams, trans)[entity][max_score_ent]
                    else:
                        transitionProb = 1/(unigrams[entity]+ len(unigrams))   
                else:
                    lexicalProb = 0
                    transitionProb = 0
                score = {tok2: {entity: max_score*lexicalProb*transitionProb}}
                backpointer = {tok2: {entity: max_score_ent}}
            else:
                if entity in getlexicalProb(train, lex_list, token_list)["<unk>"]:
                    lexicalProb = getlexicalProb(train, lex_list, token_list)["<unk>"][entity]
                    if max_score_ent in getTransitionProb(unigrams, trans)[entity]:
                        transitionProb = getTransitionProb(unigrams, trans)[entity][max_score_ent]
                    else:
                        transitionProb = 1/(unigrams[entity]+ len(unigrams))
                else:
                    lexicalProb = 0 
                    transitionProb = 0
                score = {tok2: {entity: max_score*lexicalProb*transitionProb}}
                backpointer = {tok2: {entity: max_score_ent}}
   
    ans_dict = {tok2: backpointer[tok1][max_score_ent]}   
        
print ans_dict
            
    

#finishing
                
                
            
    
    
    
    
    
    
    
    
    
    