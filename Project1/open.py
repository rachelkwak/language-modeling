import time
import compute as c
from numpy import array
from gensim.models.keyedvectors import KeyedVectors as kv

start_time = time.time()
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

model = kv.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))
print("Loading model done, Initializing")
start_time = time.time()


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

print("--- %s seconds ---" % (time.time() - start_time))
print("Initialization done, computing...")
start_time = time.time()

for k,v in analogy_words.items():
	accuracy = c.computeAccuracy(v, word_vec)
	print("Accuracy for {} is: {:.2f}%".format(k, accuracy))

print("--- %s seconds ---" % (time.time() - start_time))
"""
count = 0
analogy_task_words = {'typeof':[], 'tools':[]}
with open("analogy_task.txt") as file:
	for line in file:
		words = line.rstrip().split()
		key = 'typeof' if count < 3 else 'tools'
		analogy_task_words[key].append(words)
		[word_vec.update({i:array(model[i])}) for i in words if i not in word_vec]
		count += 1

for k,v in analogy_task_words.items():
	accuracy = c.computeAccuracy(v, word_vec)
	print("Accuracy for {} is: {:.2f}%".format(k, accuracy))
"""