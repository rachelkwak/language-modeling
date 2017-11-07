import json
import string
import sys
import nltk
import math
from collections import Counter

"""
Create iob and pos tags for contexts and questions in dataset
context_file: File to hold the tagged contexts
              Each article is separated by an empty line and the title is written at the first line for each section. 
              Each paragraph of an article is separated by an empty line. 
              For every 5 words of a paragraph, first line is the word tokens, second line is the pos tags, third line is the iob tags
questions_file: File to hold the tagged questions
                Each article is seaprated by 2 empty lines and a line with the article's title. 
                Each paragraph of an article is separated by an empty line. 
                For each question of a paragraph, first line is the word tokens, second line is the pos tags, third line is the iob tags. 
                All three lines starts with the qa['id'].
"""
def tag_dataset(dataset_file, context_file, questions_file):
  expected_version = '1.1'
  with open(dataset_file) as dataset_file:
    dataset_json = json.load(dataset_file)
    if (dataset_json['version'] != expected_version):
      print('Evaluation expects v-' + expected_version + ', but got dataset with v-' + dataset_json['version'])
    dataset = dataset_json['data']
  tagged_context_file = open(context_file, 'w')
  tagged_questions_file = open(questions_file, 'w')
  for article in dataset:
    for paragraph in article['paragraphs']:
      context = paragraph['context']
      
      # passage = sentences
      #sentences = nltk.sent_tokenize(context)
      #tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

      # passage = word chucks of length 5
      context_list = context.rstrip().split()
      word_chunks = [" ".join(context_list[i:i+5]) for i in range(0, len(context_list), 5)]
      tokenized_sentences = [nltk.word_tokenize(sent) for sent in word_chunks]

      sentences_pos = [nltk.pos_tag(sent) for sent in tokenized_sentences]
      sentences_iob = [nltk.tree2conlltags(nltk.ne_chunk(sent)) for sent in sentences_pos]
      for sen in sentences_iob:
        words = []
        pos = []
        ner = []
        for w, p, n in sen:
          words.append(w)
          pos.append(p)
          ner.append(n)
        tagged_context_file.write(" ".join(words) + "\n")
        tagged_context_file.write(" ".join(pos) + "\n")
        tagged_context_file.write(" ".join(ner) +"\n")
      
      for qa in paragraph['qas']:
        question = qa['question']
        tokenized_question = nltk.word_tokenize(question)
        question_pos = nltk.pos_tag(tokenized_question)
        question_iob = nltk.tree2conlltags(nltk.ne_chunk(question_pos))
        words = [word for (word, _, _) in question_iob]
        pos = [pos for (_, pos, _) in question_iob]
        ner = [ner for (_, _, ner) in question_iob]
        tagged_questions_file.write(qa['id'] + " " + " ".join(words) + "\n")
        tagged_questions_file.write(qa['id'] + " " + " ".join(pos) + "\n")
        tagged_questions_file.write(qa['id'] + " " + " ".join(ner) +"\n")

      tagged_questions_file.write("\n")
      tagged_context_file.write("\n")
    tagged_questions_file.write("\n")
    tagged_context_file.write("\n")


"""
Return a list data structure that contains the questions and their tags.
The outermost list is the whole dataset, second layer of list represetns each article, 
third layer the paragraphs, fourth layer the questions, and last layer list of the word token, pos tag, and iob tag for each word
"""
def get_tagged_questions(file):
  articles = []
  article = []
  paragraph = []
  with open(file, encoding="utf-8") as f:
    lines = f.readlines()
    l = 0
    while l < len(lines):
      if lines[l] == "\n":
        if not paragraph:
          articles.append(article)
          article = []
        else:
          article.append(paragraph)
        l += 1
        paragraph = []
      else:
        toks = lines[l].rstrip().split()
        pos = lines[l+1].rstrip().split()
        iob = lines[l+2].rstrip().split()
        sentence = []
        for t, p, i in zip(toks, pos, iob):
          sentence.append((t, p, i))
        paragraph.append(sentence)
        l += 3

      if l+3 == len(lines):
        toks = lines[l].rstrip().split()
        pos = lines[l+1].rstrip().split()
        iob = lines[l+2].rstrip().split()
        sentence = []
        for t, p, i in zip(toks, pos, iob):
          sentence.append((t, p, i))
        paragraph.append(sentence)
        article.append(paragraph)
        articles.append(article)
        l += 3
  return articles


"""
Return a list data structure that contains the context and their tags.
The outermost list is the whole dataset, second layer of list represetns each article, 
third layer the paragraphs, fourth layer the sentence, and last layer list of the word token, pos tag, and iob tag for each word
"""
def get_tagged_articles(file):
  articles = []
  article = []
  paragraph = []
  with open(file, encoding="utf-8") as f:
    lines = f.readlines()
    l = 0
    while l < len(lines):
      if lines[l] == "\n":
        if not paragraph:
          articles.append(article)
          article = []
        else:
          article.append(paragraph)
        l += 1
        paragraph = []
      else:
        toks = lines[l].rstrip().split()
        pos = lines[l+1].rstrip().split()
        iob = lines[l+2].rstrip().split()
        sentence = []
        for t, p, i in zip(toks, pos, iob):
          sentence.append((t, p, i))
        paragraph.append(sentence)
        l += 3
      if l+3 == len(lines):
        toks = lines[l].rstrip().split()
        pos = lines[l+1].rstrip().split()
        iob = lines[l+2].rstrip().split()
        sentence = []
        for t, p, i in zip(toks, pos, iob):
          sentence.append((t, p, i))
        paragraph.append(sentence)
        article.append(paragraph)
        articles.append(article)
        l += 3
  return articles


"""
Returns the sentence number with the highest overlapping content words with the question
"""
def overlaps(paragraph, question):
  index = 0
  amount = -1
  for sent_num, sent in enumerate(paragraph):
    count = sum([1 for word in question if word in sent])
    if count > amount:
      index = sent_num
      amount = count
  return index

"""
Returns two arrays: qc_words: list of the unique question content words
                    qc_vector: vector representation of qc_words, keeps count of the number of occurances of the content words in the question. 
The arrays corresponds to eachother via their index number. 
"""
def create_question_vector(question_content_words):
  qc_words, qc_vector = [], []
  for word in question_content_words:
    if word not in qc_words:
      qc_words.append(word)
      qc_vector.append(1) 
    else:
      qc_vector[qc_words.index(word)] += 1
  return qc_words, qc_vector


"""
  Return the words for the answer based on the answer length constraint 
"""
def create_answer(top_entities, qc_words):
  answer = []
  answer_length = 0
  beginning = ['[','(', '"', '<']
  ending = [']',')', ',', ':', ";", '?', '.', '/']
  words = []
  begin = ""
  for entity in top_entities:
    toks = [tok for tok, _, _ in entity if tok not in qc_words]
    for tok in toks:
      if answer_length < 10:
        if tok in beginning:
          begin += tok
        elif tok in ending or "'" in tok:
          if len(answer):
            answer[-1] = answer[-1] + tok
          else:
            answer.append(tok)
        else:
          answer.append(begin + tok)
          answer_length += 1
          begin = ""
      else:
        return answer
  return answer

def cosine_similarity(a, b):
    score = 0
    
    length_a = 0
    for x in a.values():
        length_a += x * x
    length_a = math.sqrt(length_a)
    
    length_b = 0
    for x in b.values():
        length_b += x * x
    length_b = math.sqrt(length_b)
    
    for word in b.keys():
        score += a[word] * b[word]
    score /= length_a * length_b
    
    return score


"""
Returns a dictionary of answers for the questions
"""
def passage_retrieval(articles, questions):
  answer_dict = {}
  # pos tags for content words
  content_tags = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'POS', 'PRP',
  'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

  for article_num, article in enumerate(questions): # loop through the questions
    for paragraph_num, paragraph in enumerate(article): # loop through paragraphs of an article

      for question in paragraph:
        # get the content words from the question and create the array for the unqiue content words and the vector representation 
        question_content_words = [tok for tok, pos, _ in question if pos in content_tags]
        question_counter = Counter(question_content_words)
        
        qc_words, qc_vector = create_question_vector(question_content_words)

        scores = []
        top_scored = [] # HOLD THE TOP N SCORED ENTITIES HERE (SHOULD BE 3 WORD CHUNKS), ordered from highest to lowest score

        # DO THE PICKING OF THE TOP K AND N SCORED ENTITIES IN HERE
        for word_chunk in articles[article_num][paragraph_num]: 
          article_counter = Counter([tok for tok, _, _ in word_chunk])
          scores.append((cosine_similarity(question_counter, article_counter), word_chunk))

        top_scored = [i[1] for i in sorted(scores, key=lambda x: x[0], reverse=True)[:3]]

        # FAKE TOP 3 SCORED ENTITES 
        #top_scored = articles[article_num][paragraph_num][:3]
        answer = create_answer(top_scored, qc_words)

        """
        # get the sentence with highest overlap in content words
        answer_sentence = articles[article_num][paragraph_num][overlaps(paragraph_content_words, question_content_words)]
        # determine which question it is and give answer accordingly
        answer = []
        question_words = [tok.lower() for tok, _, _ in question]
        if "who" in question_words:
          answer = [tok for tok, _, iob in answer_sentence if "PERSON" in iob and tok not in question_content_words]
          if not answer:
            answer = [tok for tok, _, iob in answer_sentence if "ORGANIZATION" in iob and tok not in question_content_words]
        elif "where" in question_words:
          answer = [tok for tok, _, iob in answer_sentence if "GPE" in iob and tok not in question_content_words]
        elif "when" in question_words or ("how" in question_words and "many" in question_words):
          answer = [tok for tok, pos, _ in answer_sentence if "CD" in pos] 
        elif "which" in question_words:
          answer = [tok for tok, _, iob in answer_sentence if "ORGANIZATION" in iob and tok not in question_content_words]
        if not answer:
          answer = [tok for tok, pos, _ in answer_sentence if ("NN" in pos) and tok not in question_content_words]
        """
        # add answer to the answer_dict, where the key is the question id
        answer_dict[question[0][0]] = " ".join(answer)
  return answer_dict


"""
Takes in a dictionary where the key is qa['id'] and the value is the answer to the question and outputs it to a json file
Example: ans = {"57284e9fff5b5019007da154": "are a center in", "57284e9fff5b5019007da152": "a campus located"}
"""
def output_predictions(ans, file_name):
  with open(file_name, "w") as f:
    f.write(json.dumps(ans))


def main():
  # Files for holding the tagged data
  tagged_context_file = "tagged_test_context.txt"
  tagged_questions_file = "tagged_test_questions.txt"

  # NOTE: only run this line if the tagged text files do not exist for your json file
  #tag_dataset('testing.json', tagged_context_file, tagged_questions_file)

  # Get the tagged data
  tagged_articles = get_tagged_articles(tagged_context_file)
  tagged_questions = get_tagged_questions(tagged_questions_file)


  # Get answer predictions for the questions using the tagged data
  answer_predictions = passage_retrieval(tagged_articles, tagged_questions)
  print(answer_predictions)

  # Output the answer predictions with the specified .json file
  #output_predictions(answer_predictions, "development_predictions.json")

if __name__ == '__main__':
    main()
