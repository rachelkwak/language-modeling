from __future__ import division
import sys, random
import math
from collections import defaultdict

class Ngrams(object):
	def __init__(self, text):
		self.text = text
		self.tokens = []
		self.unigrams = {"<unk>": 0}
		self.bigrams = {"<unk>":{"<unk>":0}}
		self.linestore = []

	def initTokens(self):
		with open(self.text) as file:
			for line in file:
				line = line.rstrip()
				self.tokens.append("<start>")
				self.tokens.extend(line.split())
		return self.tokens

	def storeLines(self):
		with open(self.text) as files:
			for line in file:
				line = line.rstrip()
				self.linestore.append(line)

		return self.linestore
	#if a word has never been seen before, add to unk count 1, add word as 0
	#also add it to a list of unknow words, ie the vocabulary
	#if it has been seen (and is checked by looking at the vocabulary, add to own word count
	def getUnigrams(self):
		self.initTokens()
		for token in self.tokens:
			if token in self.unigrams:
				self.unigrams[token] += 1
			else:
				self.unigrams["<unk>"] +=1
				self.unigrams[token] = 0

		return self.unigrams

	# def getUnigrams(self):
	# 	self.initTokens()
	# 	for token in self.tokens:
	# 		if token in self.unigrams:
	# 			self.unigrams[token] += 1
	# 		else:
	# 			self.unigrams[token] = 1

	# 	return self.unigrams



	# def getBigrams(self):
	# 	self.initTokens()
	# 	for word1, word2 in zip(self.tokens, self.tokens[1:]):
	# 		if word1 != ".":
	# 			if word2 in self.bigrams[word1]:
	# 				self.bigrams[word1][word2] += 1
	# 			else:
	# 				self.bigrams[word1][word2] = 1
	# 	return self.bigrams

	def getBigramsForTraining(self):
		self.initTokens()
		for word1, word2 in zip(self.tokens, self.tokens[1:]):
			if word1 != ".":
				if word1 in self.bigrams:
					if word2 in self.bigrams[word1]:
						self.bigrams[word1][word2] += 1
					else:
						self.bigrams[word1]= {"<unk>" : 1}
						self.bigrams[word1] = {word2 : 0}
				else:
					if word2 not in self.bigrams["<unk>"]:
						self.bigrams["<unk>"]["<unk>"] +=1
						self.bigrams[word1] = {word2: 0}
					else:
						self.bigrams["<unk>"][word2] += 1
						self.bigrams[word1] = {word2: 0}
		return self.bigrams

	def getProbabilityUnigrams(self):
		self.getUnigrams()
		n = len(self.tokens)
		voc = len(self.unigrams)
		return {k: ((v +5)/(n+5*voc)) for k,v in self.unigrams.items()}

	def getProbabilityBigrams(self):
		self.getUnigrams()
		self.getBigramsForTraining()
		voc = len(self.unigrams)
		return {k1: {k2: (v2+5) / (self.unigrams[k1]+ 5*voc) for k2,v2 in v1.items()} for k1,v1 in self.bigrams.items()}

def perplexityUnigram(line, ngramdict):
	l = line.rstrip().split()
	n = len(l)
	logp = 0
	for word in l:
		if word in ngramdict.keys():
			logp += math.log(1/ngramdict[word])
		else:
			logp += math.log(1/ngramdict["<unk>"])

	logperplexity = logp*(1/n)
	perplexity = math.exp(logperplexity)
	return perplexity

def perplexityBigram(line, ngramdict):
	l = line.rstrip().split()
	n = len(l)
	logp = 0
	for word1, word2 in zip(l, l[1:]):
		if word1 in ngramdict.keys():
			if word2 in ngramdict[word1].keys():
				logp += math.log(ngramdict[word1][word2])
			else:
				if "<unk>" in ngramdict[word1].keys():
					logp += math.log(ngramdict[word1]["<unk>"])
				else:
					ngramdict[word1] = {"<unk>": 1/n}
					logp += math.log(ngramdict[word1]["<unk>"])

		else: 
			if word2 in ngramdict["<unk>"].keys():
				logp += math.log(ngramdict["<unk>"][word2])
			else:
				ngramdict["<unk>"] = {"<unk>": 1/n}
				logp += math.log(ngramdict["<unk>"]["<unk>"])

	logperplexity = -1*logp*(1/n)
	perplexity = math.exp(logperplexity)
	return perplexity


def pickWeighted(ngram):
    total = sum(w for c, w in ngram.items())
    r = random.uniform(0, total)
    upto = 0
    for c, w in ngram.items():
       if upto + w > r:
          return c
       upto += w

def randomUnigram(unigram):
	word, next_word = "", ""
	sentence_length = 0
	punctuation = set([",", ".", "``", "...", "`"])
	while True:
		next_word = pickWeighted(unigram)

		# ignore start token
		while next_word == "<start>":
			next_word = pickWeighted(unigram)

		# check that first word is not punctuation
		if sentence_length == 0:
			while next_word in punctuation:
				next_word = pickWeighted(unigram)

		# end sentence with period if it has greater than five words
		if next_word == ".":
			if sentence_length > 5:
				word += "."
				break
			else:
				continue

		word = word + next_word + " "
		sentence_length += 1
	return word

def randomBigram(bigram,seed="<start>"):
	if seed == "<start>":
		word = ""
	else:
		word = seed + " "
	next_word = pickWeighted(bigram[seed])
	while next_word != ".":
		# ignore start token
		while next_word == "<start>":
			next_word = pickWeighted(bigram[next_word])

		word = word + next_word + " "
		next_word = pickWeighted(bigram[next_word])
	return word + next_word



def main():
	# pos = Ngrams("pos.txt")
	# #train
	# #unigram_prob_dict = pos.getProbabilityUnigrams()
	# print "positive perplexity unigram: "+ str(perplexityUnigram(pos, unigram_prob_dict))

	# bigram_prob_dict = pos.getProbabilityBigrams()
	# print "positive perplexity bigram: " + str(perplexityBigram(pos, bigram_prob_dict))

	# # print "pos unigram: " + randomUnigram(pos.getProbabilityUnigrams()) + "\n"
	# # print "pos bigram: " + randomBigram(pos.getProbabilityBigrams()) + "\n"
	
	# neg = Ngrams("neg.txt")

	# unigram_prob_dict = neg.getProbabilityUnigrams()
	# print "negative perplexity unigram: "+ str(perplexityUnigram(neg, unigram_prob_dict))

	# bigram_prob_dict = neg.getProbabilityBigrams()
	# print "negative perplexity bigram: " + str(perplexityBigram(neg, bigram_prob_dict))
	# print "neg unigram: " + randomUnigram(neg.getProbabilityUnigrams()) + "\n"
	# print "neg bigram: " + randomBigram(neg.getProbabilityBigrams()) + "\n"

	# print "seeded pos bigram: " + randomBigram(pos.getProbabilityBigrams(), "I") + "\n"

	#model1 - unigram positive 
	#train on training set: unigram_prob_dict
	pos = Ngrams("Train/pos.txt")
	unigram_prob_dict_pos = pos.getProbabilityUnigrams()
	bigram_prob_dict_pos = pos.getProbabilityBigrams()

	neg = Ngrams("Train/neg.txt")
	unigram_prob_dict_neg = neg.getProbabilityUnigrams()
	bigram_prob_dict_neg = neg.getProbabilityBigrams()

	#test = Ngrams("dev/pos.txt")
	with open("dev/pos.txt") as file:
			for line in file:
				perpdict = 	{"pos_unigram": perplexityUnigram(line, unigram_prob_dict_pos),
							"pos_bigram" : perplexityBigram(line, bigram_prob_dict_pos),
							"neg_unigram" : perplexityUnigram(line, unigram_prob_dict_neg),
							"neg_bigram" : perplexityBigram(line, bigram_prob_dict_neg)
							}
				with open("record.txt", "a") as newfile:
					
					newfile.write(line + min(perpdict, key = perpdict.get))



if __name__ == '__main__':
	main()

