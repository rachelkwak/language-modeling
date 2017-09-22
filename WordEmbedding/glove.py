import time
import compute as c
from numpy import array
import pandas as pd
import csv

# returns the word represented as a vector
def model(w, mod):
	try:
		return mod.loc[w].as_matrix()
	except KeyError:
		return [0.004]*100+[0.003]*100+[0.002]*100

def initWordVec(filename, mod):
	word_vec = {}
	analogy_words = {}
	categories = ['pasttense','adverb','comparative','plural','state','currency','capital','nationality']

	keys = set(['dancing danced decreasing decreased',
				'amazing amazingly apparent apparently',
				'bad worse big bigger',
				'banana bananas bird birds',
				'Chicago Illinois Houston Texas',
				'Algeria dinar Angola kwanza',
				'Athens Greece Baghdad Iraq',
				'Albania Albanian Argentina Argentinean'])

	with open(filename) as file:
		category = -1
		for line in file:
			l = line.rstrip()
			words = l.split()
			if l in keys:
				keys.remove(l)
				category += 1
				analogy_words[categories[category]] = []

			analogy_words[categories[category]].append(words)
			[word_vec.update({i:array(model(i, mod))}) for i in words if i not in word_vec]

	return analogy_words, word_vec

def printAccuracy(words, word_vec):
	for k,v in words.items():
		accuracy = c.computeAccuracy(v, word_vec)
		print("Accuracy for {} is: {:.2f}%".format(k, accuracy))

def analogyTask(filename, word_vec, mod):
	analogy_task_words = {'typeof':[], 'tools':[]}

	count = 0
	with open(filename) as file:
		for line in file:
			words = line.rstrip().split()
			key = 'typeof' if count < 3 else 'tools'
			analogy_task_words[key].append(words)
			[word_vec.update({i:array(model(i, mod))}) for i in words if i not in word_vec]
			count += 1
	return analogy_task_words, word_vec

def similarWords(word_vec, mod):
	while True:
		word = input("Enter word, type getmeout to exit: ")
		if word == "getmeout":
			break
		elif len(word) == 0:
			continue
		print("Ten most similar words to \"{}\": {}".format(word, \
			c.findSimilar(array(model(word, mod)), word_vec)), end="\n\n")

def main():
	start_time = time.time()
	print("Loading model...")
	
	mod = pd.read_table("glove/output.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

	print("--- %s seconds ---" % (time.time() - start_time))
	print("Loading model done, initializing...")
	start_time = time.time()

	# initializes word_vec with words in analogy_test.txt
	analogy_words, word_vec = initWordVec("analogy_test.txt", mod)
	
	print("--- %s seconds ---" % (time.time() - start_time))
	print("Initialization done, computing...")
	start_time = time.time()

	# print the accuracies of analogy words
	printAccuracy(analogy_words, word_vec)
	
	print("--- %s seconds ---" % (time.time() - start_time))

	# print the accuracies of analogy task
	analogy_task_words, word_vec = analogyTask("analogy_task.txt", word_vec, mod)
	printAccuracy(analogy_task_words, word_vec)

	# print ten most similar words when prompted
	similarWords(word_vec, mod)


if __name__ == '__main__':
	main()