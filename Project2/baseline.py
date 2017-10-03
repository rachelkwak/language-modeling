"""
for line in open("test.txt"):
    ## Get the next line of the metadata file, and split it into columns
    fields = line.rstrip().split("\t")
    print fields
"""
output = open("output.txt", "w")
output.write("Type,Prediction\n")
output.write("ORG,")

def check(words, pos, i):
	return words[i][0].isupper() and pos[i] == "NNP"

with open("test.txt") as f:
	while True:
		words, pos, index = [f.readline().rstrip().split() for _ in range(3)]
		i = 0
		while i < len(words):
			if check(words, pos, i):
				range_index = index[i]
				while i+1 < len(words) and check(words, pos, i+1):
					i += 1
				range_index = str(range_index) + "-" + str(index[i])
				output.write(range_index + " ")
			i += 1
		if not f.readline():
			break

output.write("\n")
output.write("MISC,\n")
output.write("PER,\n")
output.write("LOC,\n")