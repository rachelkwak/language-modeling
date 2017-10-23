from itertools import zip_longest
from collections import defaultdict
import math

class HMM():
	def __init__(self, train, test):
		self.iob_tags = ["<starten>", "B-PER", "I-PER", "B-LOC","I-LOC","B-ORG","I-ORG", \
		"B-MISC", "I-MISC", "O"]

		self.train_list = self.tokenize_train_list(train)
		self.test_list = self.tokenize_test_list(test, set([i[0] for i in self.train_list]))

		self.num_iob = len(self.iob_tags)
		self.num_tests = len(self.test_list)

		self.trained_lexical_counts = self.get_lexical_counts(self.train_list)
		self.iob_counts = self.get_iob_counts(self.iob_tags, self.train_list)
		self.bigram_transitions = self.get_bigram_transitions(self.train_list)

		self.lexical_prob = self.get_lexical_prob(self.test_list, self.iob_tags, self.trained_lexical_counts, self.iob_counts)
		self.transition_prob = self.get_transition_prob(self.iob_tags, self.bigram_transitions, self.iob_counts)

		self.T = self.viterbi(self.num_iob, self.num_tests, self.test_list, self.lexical_prob, self.transition_prob)

	def tokenize_train_list(self, file):
		train_list = []
		with open(file) as f:
			for toks, pos, iob in zip_longest(*[f]*3, fillvalue = None):
				train_list.append(("<start>", "<startp>", "<starten>"))
				for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
					train_list.append((tok,p,i))
		return train_list

	def tokenize_test_list(self, file, train):
		test_list = []
		with open(file) as f:
			for toks, pos, iob in zip_longest(*[f]*3, fillvalue = None):
				test_list.append(("<start>", "<startp>", "<starten>"))
				for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
					if tok in train:
						test_list.append((tok,p,i))
					else:
						test_list.append(("<unk>",p,i))
		return test_list

	def get_lexical_counts(self, train):
		lexical_counts = defaultdict(lambda: defaultdict(int))
		for (tok, p, i) in train:
			lexical_counts[tok][i] += 1
			lexical_counts['<unk>'][i] += 1
		return lexical_counts

	def get_iob_counts(self, iob_tags, train):
		iob_counts = defaultdict(int)
		for (_, _, iob) in train:
			iob_counts[iob] += 1
		return iob_counts

	def get_bigram_transitions(self, train):
		bigram_transitions = defaultdict(lambda: defaultdict(int))
		for (_, _, iob1), (_, _, iob2) in zip(train, train[1:]):
			bigram_transitions[iob1][iob2] += 1
		return bigram_transitions

	def get_lexical_prob(self, test, iob_tags, lex_counts, iob_counts):
		lex_prob = defaultdict(lambda: defaultdict(float))
		for word, _, _ in test:
			for iob in iob_tags:
				try:
					lex_prob[word][iob] = math.log(float(lex_counts[word][iob]) / iob_counts[iob])
				except ValueError:
					lex_prob[word][iob] = -100
		return lex_prob

	def get_transition_prob(self, iob_tags, bigram_transitions, iob_counts):
		transition_prob = defaultdict(lambda: defaultdict(float))
		for iob1 in iob_tags:
			for iob2 in iob_tags:
				try:
					transition_prob[iob1][iob2] = math.log(float(bigram_transitions[iob2][iob1]) / iob_counts[iob2])
				except ValueError:
					transition_prob[iob1][iob2] = math.log(1/(float(len(iob_counts)+iob_counts[iob2])))
		return transition_prob

	def viterbi(self, num_iob, num_tests, test_list, lexical_prob, transition_prob):
		score = [[0 for i in range(num_tests)] for _ in range(num_iob)]
		bptr = [[0 for i in range(num_tests)] for _ in range(num_iob)]
		T = [0 for _ in range(num_tests)]

		for i in range(num_iob):
			score[i][0] = transition_prob[self.iob_tags[i]]["<starten>"] + lexical_prob["<start>"][self.iob_tags[i]] 

		for t in range(1, num_tests):
			for i in range(num_iob):
				max_score = -float('inf')
				max_index = 0

				for j in range(num_iob):
					prev_max = max_score
					max_score = max(score[j][t-1] + transition_prob[self.iob_tags[i]][self.iob_tags[j]], max_score)

					if max_score != prev_max:
						max_index = j

				score[i][t] = max_score + lexical_prob[test_list[t][0]][self.iob_tags[i]]
				bptr[i][t] = max_index

		max_T_index = 0
		max_T = -float('inf')

		for i in range(num_iob):
			prev = max_T
			max_T = max(score[i][num_tests-1], max_T)
			if prev != max_T:
				max_T_index = i

		T[num_tests-1] = max_T_index

		for i in range(num_tests-2, -1, -1):
			T[i] = bptr[T[i+1]][i+1]

		return [self.iob_tags[i] for i in T]

	def get_indicies(self):
		return [ind for _, _, ind in self.test_list]# if ind != "<starten>"]

	def get_iob_predictions(self):
		return [iob for iob in self.T]# if iob != "<starten>"]

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

def main():
	hmm = HMM("train.txt", "test.txt")
	"""
	for word, iob in zip(hmm.test_list, hmm.T):
		print(word, iob)
	"""
	"""
	ind = hmm.get_indicies()
	for pos, i in enumerate(ind):
		if pos != i:
			print(pos, i)
	"""
	ind = hmm.get_indicies()
	iob = hmm.get_iob_predictions()
	print(len(ind), ind.count("<starten>"))
	print(len(iob), iob.count("<starten>"))

	for a, b in zip(ind, iob):
		if (a == "<starten>" or b == "<starten>") and a != b:
			print(a,b)

	print(ind.index("1571"))
	print(hmm.test_list[1701], hmm.T[1701])
	"""
	org_pred, misc_pred, per_pred, loc_pred = test_entity_index(hmm.get_iob_predictions(), hmm.get_indicies())


	# output the results in file named output.txt
	output = open("output.txt", "w")
	output.write("Type,Prediction\n")
	output.write("ORG," + " ".join(org_pred) + "\n")
	output.write("MISC," + " ".join(misc_pred) + "\n")
	output.write("PER," + " ".join(per_pred) + "\n")
	output.write("LOC," + " ".join(loc_pred))
	"""


if __name__ == '__main__':
	main()
