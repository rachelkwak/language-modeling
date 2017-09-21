from numpy import dot, array, array_equal
from numpy.linalg import norm 

# compute the cosine similarity of two vectors
def cosine(u,v):
	return dot(u,v) / (norm(u) * norm(v))

# check if vector is equal to a vector in the matrix
def checkArray(vec, matrx):
	return all([not array_equal(vec, i) for i in matrx])

# compute the cosine similarities of a word analogy and return the max
def findWord(word_vec, matrx):
	excl_vec = {k: word_vec[k] for k in word_vec.keys() if checkArray(word_vec[k], matrx)}
	diff = matrx[1] - matrx[0] + matrx[2]
	return max(excl_vec.keys(), key=lambda k: cosine(diff, excl_vec[k]))

def computeAccuracy(analogy_words, word_vec):
	accuracy = 0
	for analogy in analogy_words:
		matrx = [word_vec[i] for i in analogy[:3]]
		word = findWord(word_vec, matrx)
		if word == analogy[3]:
			accuracy += 1
	return 100 * accuracy / len(analogy_words)