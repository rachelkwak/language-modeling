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
		""" 
		Converts the training file into a list of (token, POS tag, IOB tag) tuples 

		"""
		train_list = []
		with open(file) as f:
		    for toks, pos, iob in zip_longest(*[f]*3, fillvalue = None):
		        train_list.append(("<start>", "<startp>", "<starten>"))
		        for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
		            train_list.append((tok,p,i))
		return train_list

	def tokenize_test_list(self, file, train):
		"""
		Converts the test file into a list of (token, POS, index) tuples. If token is not in the 
		training set, it is stored as <unk>

		"""
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
		""" 
		Gets the count of each word for each IOB tag in the training set. Each IOB tag is also counted 
		under <unk> to collect the distribution of IOB tags.
		
		"""
		lexical_counts = defaultdict(lambda: defaultdict(int))
		for (tok, p, i) in train:
			lexical_counts[tok][i] += 1
			lexical_counts['<unk>'][i] += 1
		return lexical_counts

	def get_iob_counts(self, iob_tags, train):
		""" Gets the count of each IOB tag in the training set """
		iob_counts = defaultdict(int)
		for (_, _, iob) in train:
			iob_counts[iob] += 1
		return iob_counts

	def get_bigram_transitions(self, train):
		""" Gets the count of two IOB tags occuring consecutively """
		bigram_transitions = defaultdict(lambda: defaultdict(int))
		for (_, _, iob1), (_, _, iob2) in zip(train, train[1:]):
			bigram_transitions[iob1][iob2] += 1
		return bigram_transitions

	def get_lexical_prob(self, test, iob_tags, lex_counts, iob_counts):
		""" 
		Calculates the probability of P(word | iob) for each word and IOB tag in the test set and
		stores it in a dictionary

		"""
		lex_prob = defaultdict(lambda: defaultdict(float))
		for word, _, _ in test:
			for iob in iob_tags:
				try:
					lex_prob[word][iob] = math.log(float(lex_counts[word][iob]) / iob_counts[iob])
				except ValueError:
					lex_prob[word][iob] = -100
		return lex_prob

	def get_transition_prob(self, iob_tags, bigram_transitions, iob_counts):
		"""	
		Calculates the probability of P(iob1 | iob2) for each IOB tag and stores it in a dictionary

		"""
		transition_prob = defaultdict(lambda: defaultdict(float))
		for iob1 in iob_tags:
			for iob2 in iob_tags:
				try:
					transition_prob[iob1][iob2] = math.log(float(bigram_transitions[iob2][iob1]) / iob_counts[iob2])
				except ValueError:
					transition_prob[iob1][iob2] = math.log(1/(float(len(iob_counts)+iob_counts[iob2])))
		return transition_prob

	def viterbi(self, num_iob, num_tests, test_list, lexical_prob, transition_prob):
		""" Gets the list of predicted IOB tags of the words in a test list with the viterbi algorithm """

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


def main():
	hmm = HMM("train.txt", "test.txt")
	for word, iob in zip(hmm.test_list, hmm.T):
		print(word, iob)


if __name__ == '__main__':
	main()