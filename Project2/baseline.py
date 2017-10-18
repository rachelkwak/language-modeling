def check(words, pos, i):
	return words[i][0].isupper() and pos[i] == "NNP"

all_ranges = []
with open("test.txt") as f:
	while True:
		words, pos, index = [f.readline().rstrip().split() for _ in range(3)]
		if not words or not pos or not index:
			break
		i = 0
		while i < len(words):
			if check(words, pos, i):
				range_index = index[i]
				while i+1 < len(words) and check(words, pos, i+1):
					i += 1
				all_ranges.append(str(range_index) + "-" + str(index[i]))
			i += 1


output = open("output.txt", "w")
output.write("Type,Prediction\n")
output.write("ORG,")
for i in all_ranges[:len(all_ranges)/4-1]:
    output.write(i + " ")
output.write(all_ranges[len(all_ranges)/4-1])
output.write("\n")

output.write("MISC,")
for i in all_ranges[len(all_ranges)/4:len(all_ranges)*2/4-1]:
    output.write(i + " ")
output.write(all_ranges[len(all_ranges)/4-1])
output.write("\n")

output.write("PER,")
for i in all_ranges[len(all_ranges)*2/4:len(all_ranges)*3/4-1]:
    output.write(i + " ")
output.write(all_ranges[len(all_ranges)*3/4-1])
output.write("\n")

output.write("LOC,")
for i in all_ranges[len(all_ranges)*3/4:len(all_ranges)-1]:
    output.write(i + " ")
output.write(all_ranges[len(all_ranges)-1])
