import time
import compute as c
from numpy import array
from gensim.models.keyedvectors import KeyedVectors as kv


def initWordVec(filename, model):
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

	with open("analogy_test.txt") as file:
		category = -1
		for line in file:
			l = line.rstrip()
			words = l.split()
			if l in keys:
				keys.remove(l)
				category += 1
				analogy_words[categories[category]] = []
			analogy_words[categories[category]].append(words)
			[word_vec.update({i:array(model[i])}) for i in words if i not in word_vec]

	return analogy_words, word_vec


def printAccuracy(words, word_vec):
	for k,v in words.items():
		accuracy = c.computeAccuracy(v, word_vec)
		print("Accuracy for {} is: {:.2f}%".format(k, accuracy))


def analogyTask(filename, word_vec, model):
	count = 0
	analogy_task_words = {'typeof':[], 'tools':[]}
	with open("analogy_task.txt") as file:
		for line in file:
			words = line.rstrip().split()
			key = 'typeof' if count < 3 else 'tools'
			analogy_task_words[key].append(words)
			[word_vec.update({i:array(model[i])}) for i in words if i not in word_vec]
			count += 1
	return analogy_task_words, word_vec

def similarWords(word_vec, model):
	while True:
		word = input("Enter word, type getmeout to exit: ")
		if word == "getmeout":
			break
		elif len(word) == 0:
			continue
		try:
			print("Ten most similar words to \"{}\": {}".format(word, \
				c.findSimilar(array(model[word]), word_vec)), end="\n\n")
		except KeyError:
			print("Sorry, please enter another word: ")

def main():
	start_time = time.time()
	print("Loading model...")

	model = kv.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

	print("--- %s seconds ---" % (time.time() - start_time))
	print("Loading model done, initializing...")
	start_time = time.time()

	# initializes word_vec with words in analogy_test.txt
	analogy_words, word_vec = initWordVec("analogy_test.txt", model)
	
	print("--- %s seconds ---" % (time.time() - start_time))
	print("Initialization done, computing...")
	start_time = time.time()

	# print the accuracies of analogy words
	printAccuracy(analogy_words, word_vec)

	print("--- %s seconds ---" % (time.time() - start_time))
	
	# print the accuracies of analogy task
	analogy_task_words, word_vec = analogyTask("analogy_task.txt", word_vec, model)
	printAccuracy(analogy_task_words, word_vec)

	# print ten most similar words when prompted
	similarWords(word_vec, model)


if __name__ == '__main__':
	main()