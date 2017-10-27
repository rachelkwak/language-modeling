from itertools import izip_longest
import operator

def tag_dataset(filename):

	orig_train = []
	with open(filename) as f:
	    # for toks, pos, iob in izip_longest(*[f]*3, fillvalue = None):
	    # 	paragraphs = []
	    # 	articles = []
	    #     for tok, p, i in zip(tok.split("/n/n"), pos.split("/n/n"), iob.split("/n/n")):
	    #         articles.append((toks,p,i))
	    #         for t_p, p_p, i_p in zip(toks.rstrip().split(), p.rstrip().split(), i.rstrip().strip()):
	    #         	paragraphs.append((t_p, p_p, i_p))
	    # 		orig_train.append(paragraphs)

	#return orig_train
		data = f.read()
		
		dataset =[]
		articles = data.split("\n\n\n")

		for article in articles:
			paragraphs = []
			paragraph = article.split("\n\n")
			paragraphs.append(paragraph)
			dataset.append(paragraphs)

		return dataset

def createtuples(divided_data):

	original_tup_list =[]

	for articles in divided_data:
		art =[]
		for para in articles:
			par = []
			for toks, pos, iob in izip_longest(*[para]*3, fillvalue = None):

				for tok, p, i in zip(toks.rstrip().split(), pos.rstrip().split(), iob.rstrip().split()):
					par.append((tok,p,i))

			art.append(par)
        original_tup_list.append(art)
	
	return original_tup_list


def baseline():
	dev = tag_dataset("tagged_development_context.txt")
	dev_context = createtuples(dev)
	#[[[p1_tagged], [p11_tagged]..],
	#[[p2_tagged], [p22_tagged]..],
	#[[q1_tagged], [q11_tagged]..[qn_tagged]]]

	ques = tag_dataset("tagged_development_questions.txt")
	ques_context = createtuples(ques)

	#check if questions are who, where, when
	#pick subject of the question
	#find subject in the context
	#look at NE of context, 
	##find closest, person, location or time
	##if they don;t exist
	##find closest noun
	ans_dict = {}

	for i in range(len(ques_context)):					#articles
		for j in range(len(ques_context[i])): 				#paragraphs
			counter = ''
			qid =''
			for word,pos,iob in range(len(ques_context[i][j])):		#individual words
				print word

				#get question id
				if word == pos and pos == iob:
					qid = word

				#check for the presence of who, where or when in the question
				if word.lower() == 'who':
					counter ='who'

				if word.lower() == 'where':
					counter = 'where'

				if word.lower() == 'when':
					counter = 'when'
			
			if counter == 'who':
				#find object of the question (assuming it is the noun from the end)
				noun = []
				for word,pos,iob in range(len(ques_context[i][j])):
					if pos.find('NN') != -1:
						noun.append(word)						#list of nounds in the question

				subject = noun[len(noun) -1]					#last noun is the subject

				person = []
				#find person in context 
				for word,pos,iob in range(len(dev_context[i][j])):
					if iob.find('PERSON') != -1:
						person.append(word)						#list of people in the paragraph

				ans_dict[qid] = " ".join(person)				#return list of all people seperated by a space 


			if counter == 'where':
				#find object of the question (assuming it is the noun from the end)
				noun = []
				for word,pos,iob in range(len(ques_context[i][j])):
					if pos.find('NN') != -1:
						noun.append(word)						#list of nounds in the question

				subject = noun[len(noun) -1]					#last noun is the subject

				location = []
				#find location in context 
				for word,pos,iob in range(len(dev_context[i][j])):
					if iob.find('LOCATION') != -1:
						location.append(word)						#list of locations in the paragraph

				ans_dict[qid] = " ".join(location)				#return list of all locations seperated by a space 

			if counter == 'when':
				#find object of the question (assuming it is the noun from the end)
				noun = []
				for word,pos,iob in range(len(ques_context[i][j])):
					if pos.find('NN') != -1:
						noun.append(word)						#list of nounds in the question

				subject = noun[len(noun) -1]					#last noun is the subject

				time = []
				#find location in context 
				for word,pos,iob in range(len(dev_context[i][j])):
					if iob.find('TIME') != -1 or iob.find('DATE'):
						time.append(word)						#list of locations in the paragraph

				ans_dict[qid] = " ".join(time)				#return list of all locations seperated by a space

			else:
				#do
				continue


	return ans_dict


def main():
	return baseline()

if __name__ == '__main__':
    print main()
