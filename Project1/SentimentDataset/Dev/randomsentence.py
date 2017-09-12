from __future__ import division
import sys, random
from collections import defaultdict

class Ngrams(object):
	def __init__(self, text):
		self.text = text
		self.tokens = []
		self.unigrams = {}
		self.bigrams = defaultdict(dict)

	def initTokens(self):
		with open(self.text) as file:
			for line in file:
				line = line.rstrip()
				self.tokens.append("<start>")
				self.tokens.extend(line.split())
		return self.tokens

	def getUnigrams(self):
		self.initTokens()
		for token in self.tokens:
			if token in self.unigrams:
				self.unigrams[token] += 1
			else:
				self.unigrams[token] = 1
		return self.unigrams

	def getBigrams(self):
		self.initTokens()
		for word1, word2 in zip(self.tokens, self.tokens[1:]):
			if word1 != ".":
				if word2 in self.bigrams[word1]:
					self.bigrams[word1][word2] += 1
				else:
					self.bigrams[word1][word2] = 1
		return self.bigrams

	def getProbabilityUnigrams(self):
		self.getUnigrams()
		return {k: v / len(self.tokens) for k,v in self.unigrams.items()}

	def getProbabilityBigrams(self):
		self.getUnigrams()
		self.getBigrams()
		return {k1: {k2: v2 / self.unigrams[k1] for k2,v2 in v1.items()} for k1,v1 in self.bigrams.items()}

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
	pos = Ngrams("pos.txt")
	print "pos unigram: " + randomUnigram(pos.getProbabilityUnigrams()) + "\n"
	print "pos bigram: " + randomBigram(pos.getProbabilityBigrams()) + "\n"
	
	neg = Ngrams("neg.txt")
	print "neg unigram: " + randomUnigram(neg.getProbabilityUnigrams()) + "\n"
	print "neg bigram: " + randomBigram(neg.getProbabilityBigrams()) + "\n"

	print "seeded pos bigram: " + randomBigram(pos.getProbabilityBigrams(), "I") + "\n"

if __name__ == '__main__':
	main()

