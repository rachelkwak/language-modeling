from itertools import zip_longest
from collections import defaultdict
import math, random

class HMM():
	def __init__(self, train, test = "NULL"):
		self.iob_tags = ["<starten>", "B-PER", "I-PER", "B-LOC","I-LOC","B-ORG","I-ORG", \
		"B-MISC", "I-MISC", "O"]

		if test != "NULL":
			self.train_list = self.tokenize_train_list(train)
			self.test_list = self.tokenize_test_list(test, set([i[0] for i in self.train_list]))
		else:
			self.train_indicies = []
			self.test_indicies = []
			self.train_list, self.test_list = self.tokenize_train_test_list(train)

		self.num_iob = len(self.iob_tags)
		self.num_tests = len(self.test_list)

		self.trained_lexical_counts = self.get_lexical_counts(self.train_list)
		self.iob_counts = self.get_iob_counts(self.iob_tags, self.train_list)
		self.bigram_transitions = self.get_bigram_transitions(self.train_list)
		self.trigram_transitions = self.get_trigram_transitions(self.train_list)

		self.lexical_prob = self.get_lexical_prob(self.test_list, self.iob_tags, self.trained_lexical_counts, self.iob_counts)

		self.k = 1.0 # K-value for smoothing
		self.bigram_transition_prob = self.get_bigram_transition_prob(self.iob_tags, self.bigram_transitions, self.iob_counts)
		self.trigram_transition_prob = self.get_trigram_transition_prob(self.iob_tags, self.trigram_transitions, self.bigram_transitions, self.iob_counts)
		self.bT = self.bigram_viterbi(self.num_iob, self.num_tests, self.test_list, self.lexical_prob, self.bigram_transition_prob)
		self.tT = self.trigram_viterbi(self.num_iob, self.num_tests, self.test_list, self.lexical_prob, self.trigram_transition_prob)


	def chunk_sampling(self, content):
		"""
		Spilts the training data into 10 portion  
		Takes the last 10% portion of the training data as the test set and the rest as as training set
		"""
		lines = int(int(len(content)/3*.9))*3
		self.train_indicies = range(lines)
		self.test_indicies = range(lines, len(content))

	def random_sampling(self, content):
		"""
		Randomly takes 90% of training data by line as training set and the rest as test set
		"""
		lines = random.sample(range(int(len(content)/3)), int(int(len(content)/3*.9)))
		for i in range(int(len(content)/3)):
			if i in lines:
				self.train_indicies.append(i*3)
				self.train_indicies.append(i*3+1)
				self.train_indicies.append(i*3+2)
			else:
				self.test_indicies.append(i*3)
				self.test_indicies.append(i*3+1)
				self.test_indicies.append(i*3+2)

	def tokenize_train_test_list(self, file):
		""" 
		Converts the training file into two lists of (token, POS tag, IOB tag) tuples.
		First list is the training set and second list is the test set

		"""
		train_list = []
		test_list = []
		train_toks = []

		with open(file) as f:
			content = f.read().splitlines()

		# Choose sampling 
		# Chunk Sampling or Randomied Sampling
		# The first list contains first 90% of data from the file and the second list contains the remaining 10%.
		self.chunk_sampling(content)
		#self.random_sampling(content)

		c_train = [' '.join(content[i].split()) for i in self.train_indicies]
		c_test = [' '.join(content[i].split()) for i in self.test_indicies]

		for i in range(0, len(c_train), 3):
			train_list.append(("<start>", "<startp>", "<starten>"))
			for tok, p, i in zip(c_train[i].rstrip().split(), c_train[i+1].rstrip().split(), c_train[i+2].rstrip().split()):
				train_list.append((tok,p,i))
				train_toks.append(tok)
		for i in range(0, len(c_test), 3):
			for tok, p, i in zip(c_test[i].rstrip().split(), c_test[i+1].rstrip().split(), c_test[i+2].rstrip().split()):
				if tok in train_toks:
					test_list.append((tok,p,i))
				else:
					test_list.append(("<unk>",p,i))
		return train_list, test_list


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
		bigram_transitions = defaultdict(int)
		for (_, _, iob1), (_, _, iob2) in zip(train, train[1:]):
			if iob2 != "<starten>":
				bigram_transitions[(iob1, iob2)] += 1
		return bigram_transitions


	def get_trigram_transitions(self, train):
		""" Gets the count of three IOB tags occuring consecutively """
		trigram_transitions = defaultdict(int)
		for (_, _, iob1), (_, _, iob2), (_, _, iob3) in zip(train, train[1:], train[2:]):
			if iob3 != "<starten>":
				trigram_transitions[(iob1, iob2, iob3)] += 1
		return trigram_transitions

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

	def get_bigram_transition_prob(self, iob_tags, bigram_transitions, iob_counts):
		"""	
		Calculates the probability of P(iob1 | iob2) for each IOB tag and stores it in a dictionary

		"""
		transition_prob = defaultdict(float)
		for iob1 in iob_tags:
			for iob2 in iob_tags:
				try:
					transition_prob[(iob1, iob2)] = math.log(float(bigram_transitions[(iob2,iob1)]) / iob_counts[iob2])
				except ValueError:
					transition_prob[(iob1, iob2)] = math.log(self.k/(float(len(iob_counts)+iob_counts[iob2])))
		return transition_prob

	def get_trigram_transition_prob(self, iob_tags, trigram_transitions, bigram_transitions, iob_counts):
		"""	
		Calculates the probability of P(iob3 | iob1 iob2) for each IOB tag and stores it in a dictionary

		"""
		transition_prob = defaultdict(float)
		for iob1 in iob_tags:
			for iob2 in iob_tags:
				for iob3 in iob_tags:
					try:
						transition_prob[(iob3, iob1, iob2)] = math.log(float(trigram_transitions[(iob1, iob2, iob3)]) / bigram_transitions[(iob1, iob2)])
					except:
						transition_prob[(iob3, iob1, iob2)] = math.log(0.2*bigram_transitions[(iob3, iob2)]/iob_counts[iob3] + 0.1*iob_counts[iob3]/sum(iob_counts.values()))

		return transition_prob

	def bigram_viterbi(self, num_iob, num_tests, test_list, lexical_prob, transition_prob):
		""" Gets the list of predicted IOB tags of the words in a test list with the viterbi algorithm """

		score = [[0 for i in range(num_tests)] for _ in range(num_iob)]
		bptr = [[0 for i in range(num_tests)] for _ in range(num_iob)]
		T = [0 for _ in range(num_tests)]

		for i in range(num_iob):
			score[i][0] = transition_prob[(self.iob_tags[i], "<starten>")] + lexical_prob["<start>"][self.iob_tags[i]] 
		
		for t in range(1, num_tests):
			for i in range(num_iob):
				max_score = -float('inf')
				max_index = 0

				for j in range(num_iob):
					prev_max = max_score
					max_score = max(score[j][t-1] + transition_prob[(self.iob_tags[i], self.iob_tags[j])], max_score)

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

	def trigram_viterbi(self, num_iob, num_tests, test_list, lexical_prob, transition_prob):
		""" Gets the list of predicted IOB tags of the words in a test list with the viterbi algorithm """
		score = [[[0 for _ in range(num_tests)] for _ in range(num_iob)] for _ in range(num_iob)]
		bptr = [[[0 for _ in range(num_tests)] for _ in range(num_iob)] for _ in range(num_iob)]
		T = [0 for _ in range(num_tests)]

		for i in range(num_iob):
			for j in range(num_iob):
				score[i][j][0] = -10

		for t in range(1, num_tests):
			for i in range(num_iob):
				for j in range(num_iob):
					max_score = -float('inf')
					max_index = 0

					for k in range(num_iob):
						prev_max = max_score
						max_score = max(score[k][i][t-1] + transition_prob[(self.iob_tags[j], self.iob_tags[k], self.iob_tags[i])], max_score)

						if max_score != prev_max:
							max_index = k

					score[i][j][t] = max_score + lexical_prob[test_list[t][0]][self.iob_tags[j]]
					bptr[i][j][t] = max_index
		max_T_i = 0
		max_T_j = 0
		max_T = -float('inf')
		
		for i in range(num_iob):
			for j in range(num_iob):
				prev_T = max_T
				max_T = max(score[i][j][num_tests-1], max_T)
				if prev_T != max_T:
					max_T_i = i
					max_T_j = j

		T[num_tests-1] = max_T_j
		T[num_tests-2] = max_T_i

		for i in range(num_tests-3, -1, -1):
			T[i] = bptr[T[i+1]][T[i+2]][i+2]

		return [self.iob_tags[i] for i in T]

	def get_indicies(self):
		"""
		Gets the index for each word token
		"""
		return [ind for _, _, ind in self.test_list if ind != "<starten>"]

	def get_bigram_iob_predictions(self):
		"""
		Gets the IOB tag prediction for each word token
		"""
		return [iob for iob in self.bT if iob != "<starten>"]

	def get_trigram_iob_predictions(self):
		"""
		Gets the IOB tag prediction for each word token
		"""
		return [iob for iob in self.tT if iob != "<starten>"]

def entity_index(iobs):
	"""
	Takes in iob tags of word tokens as a list of lists (each inner list is a sentence)
	The index of the word token in the corpus should correspond with the index of its iob tag in the flatten list of iobs
	Returns the list of entities for each type: ORG, MISC, PER, and LOC; in this order
	"""
	org, misc, per, loc = [], [], [], []
	ind = 0
	while ind < len(iobs):
		if iobs[ind] != "O":
			type = iobs[ind]
			range_ind = ind
			while ind+1 < len(iobs) and iobs[ind+1] != type and type[type.index('-')+1:] in iobs[ind+1]:
				ind += 1
			irange = str(range_ind) + "-" + str(ind)
			if "ORG" in type:
				org.append(irange)
			elif "MISC" in type:
				misc.append(irange)
			elif "PER" in type:
				per.append(irange)
			else:
				loc.append(irange)
		ind += 1
	return org, misc, per, loc

def test_entity_index(iobs, indicies):
	"""
	Similar to entity_index except the positions for word token are provided and used in listing of entities
	"""
	org, misc, per, loc = [], [], [], []
	ind = 0
	while ind < len(iobs):
		if iobs[ind] != "O":
			type = iobs[ind]
			range_ind = ind
			while ind+1 < len(iobs) and iobs[ind+1] != type and type[type.index('-')+1:] in iobs[ind+1]:
				ind += 1
			irange = str(range_ind) + "-" + str(indicies[ind])
			if "ORG" in type:
				org.append(irange)
			elif "MISC" in type:
				misc.append(irange)
			elif "PER" in type:
				per.append(irange)
			else:
				loc.append(irange)
		ind += 1
	return org, misc, per, loc

def calculate_measures(org_gold, misc_gold, per_gold, loc_gold, iob_predict, model_describe):
	"""
	Calculates the precision, recall, and f1-score at entity level and prints out the results
	"""
	org_pred, misc_pred, per_pred, loc_pred = entity_index(iob_predict)
    
	# accuracy measures
	true_positive = float(sum([1 for org in org_gold if org in org_pred])
                          + sum([1 for misc in misc_gold if misc in misc_pred])
                          + sum([1 for per in per_gold if per in per_pred])
                          + sum([1 for loc in loc_gold if loc in loc_pred]))
        
	gold = len(org_gold) + len(misc_gold) + len(per_gold) + len(loc_gold)
	pred = len(org_pred) + len(misc_pred) + len(per_pred) + len(loc_pred)

	print("\n" + model_describe)
	print("Percision:  %0.5f" % (true_positive/pred))
	print("Recall:  %0.5f" % (true_positive/gold))
	print("F1-score:  %0.5f" % (2*true_positive/(pred+gold)))

def main():
	
	# training on validation set
	hmm_valid = HMM("train.txt")
	# accuracy of model
	org_true, misc_true, per_true, loc_true = entity_index(hmm_valid.get_indicies())
	calculate_measures(org_true, misc_true, per_true, loc_true, hmm_valid.get_trigram_iob_predictions(), "HMM")

	# on test data 
	hmm = HMM("train.txt", "test.txt")
	
	org_pred, misc_pred, per_pred, loc_pred = test_entity_index(hmm.get_trigram_iob_predictions(), hmm.get_indicies())
	# output the results in file named output.txt
	output = open("output.txt", "w")
	output.write("Type,Prediction\n")
	output.write("ORG," + " ".join(org_pred) + "\n")
	output.write("MISC," + " ".join(misc_pred) + "\n")
	output.write("PER," + " ".join(per_pred) + "\n")
	output.write("LOC," + " ".join(loc_pred))


	#for word, iob in zip(hmm.test_list, hmm.T):
	#	print(word, iob)


if __name__ == '__main__':
	main()
