import json
import string
import sys
import nltk

"""
Create iob and pos tags for contexts and questions in dataset
context_file: File to hold the tagged contexts
              Each article is separated by an empty line and the title is written at the first line for each section. 
              Each paragraph of an article is separated by an empty line. 
              For each sentence of a paragraph, first line is the word tokens, second line is the pos tags, third line is the iob tags
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
      sentences = nltk.sent_tokenize(context)
      tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
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
  with open(file) as f:
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
  with open(file) as f:
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
Returns a dictionary of answers for the questions
"""
def baseline(articles, questions):
  answer_dict = {}
  # pos tags for content words
  content_tags = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'POS', 'PRP',
  'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

  for article_num, article in enumerate(questions): # loop through the articles
    for paragraph_num, paragraph in enumerate(article): # loop through paragraphs of an article
      
      # get the content words in the paragraph, where each list of content words is from the same sentence
      paragraph_content_words = []
      for sent in articles[article_num][paragraph_num]: 
        sentence = [tok for tok, pos, _, in sent if pos in content_tags]
        paragraph_content_words.append(sentence)

      for question in paragraph:
        # get the content words from the question 
        question_content_words = [tok for tok, pos, _ in question if pos in content_tags]
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
  tagged_context_file = "tagged_development_context.txt"
  tagged_questions_file = "tagged_development_questions.txt"

  # NOTE: only run this line if the tagged text files do not exist for your json file
  tag_dataset('development.json', tagged_context_file, tagged_questions_file)

  # Get the tagged data
  tagged_articles = get_tagged_articles(tagged_context_file)
  tagged_questions = get_tagged_questions(tagged_questions_file)

  # Get answer predictions for the questions using the tagged data
  answer_predictions = baseline(tagged_articles, tagged_questions)

  # Output the answer predictions with the specified .json file
  output_predictions(answer_predictions, "development_predictions.json")

if __name__ == '__main__':
    main()
