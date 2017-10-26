import json
import string
import sys
import nltk

def main():
    expected_version = '1.1'
    with open('training.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version + ', but got dataset with v-' + dataset_json['version'])
        dataset = dataset_json['data']

    # organized where each document has title, paragraphs
    # paragraphs has context, qas
    # qas has question, id, answers
    # answers has text, answer_start
    # looping through each article
    for article in dataset:
        # print(article['title'])
        for paragraph in article['paragraphs']:
            #print(paragraph['context'])
            for qa in paragraph['qas']:
                #print(qa['question'], qa['id']
                for ans in qa['answers']:
                    print(ans['text'], ans['answer_start'])
                print("\n")
                

def baseline(json_data):
    context = ""
    question =""
    ans_arr =[]
    
    for article in json_data:
        for paragraph in article['paragraphs']:
            ans_arr =[]
            context = paragraphs['context']
            sentences = nltk.sent_tokenize(context)
            for qas in paragraphs['qas']:
               question = qas['question']
               
               #pos tagging for the question
               #divide the questions into who, what,where et all
               #if a question contains 'who'
               #identify subject
               ##for senetnce in context token array
               ###if subject in sentence
               ####NER sentence
               ####ans = person from NER
               
               pos_question = nltk.pos_tag(nltk.word_tokenize(question))
               iob_question = nltk.tree2conlltags(nltk.ne_chunk(pos_question))
               for word,pos in pos_question:
                   if pos == 'NNP':
                       for sentence in sentences:
                           if sentence.find(word) != -1:
                               sentence_pos = nltk.pos_tag(sentence)
                               sentence_iob = nltk.tree2conlltags(nltk.ne_chunk(sentence_pos)
                               
                               #person
                               if question.find('who') != -1:
                                   noun = []
                                   person = []
                                   
                                   for word, pos_tag, iob in sentence_iob:
                                       if iob == 'B-PERSON' or iob =='I-PERSON':
                                           person.append(iob)
                                       else:
                                           if pos_tag == 'NNP':
                                               noun.append(pos_tag)
                                    if len(person) != 0:
                                        ans_arr.append(person)
                                    else:
                                        ans_arr.append(noun)
                                
                                #location
                                if question.find('where') != -1:
                                    noun = []
                                    location = []
                                   
                                   for word, pos_tag, iob in sentence_iob:
                                       if iob == 'B-LOCATION' or iob =='I-LOCATION':
                                           location.append(iob)
                                       else:
                                           if pos_tag == 'NNP':
                                               noun.append(pos_tag)
                                    if len(person) != 0:
                                        ans_arr.append(location)
                                    else:
                                        ans_arr.append(noun)
                                        
                                #when
                                if question.find('when') != -1:
                                    noun = []
                                    time = []
                                   
                                   for word, pos_tag, iob in sentence_iob:
                                       if iob == 'B-DATE' or iob =='I-DATE' 
                                       or iob == 'B-TIME' or 'B-TIME':
                                           time.append(iob)
                                       else:
                                           if pos_tag == 'NNP':
                                               noun.append(pos_tag)
                                    if len(person) != 0:
                                        ans_arr.append(time)
                                    else:
                                        ans_arr.append(noun)
                                        
                                else:
                                    noun = []
                                    for word, pos_tag, iob in sentence_iob:
                                        if pos_tag == 'NNP':
                                            noun.append(pos_tag)
                                            
                                    ans_arr.append(nounf)
                                    
               


if __name__ == '__main__':
    main()
"""
training file
{u'title': u'Pub',
u'paragraphs':[
               {u'context': u'A pub /p\u028cb/, or public house is, despite its name, a private house, but is called a public house because it is licensed to sell alcohol to the general public. It is a drinking establishment in Britain, Ireland, New Zealand, Australia, Canada, Denmark and New England. In many places, especially in villages, a pub can be the focal point of the community. The writings of Samuel Pepys describe the pub as the heart of England.',
               u'qas': [{u'question': u'What is a pub licensed to sell?', u'id': u'56dede3c3277331400b4d784', u'answers': [{u'text': u'it is licensed to sell alcohol', u'answer_start': 105}]},
                        {u'question': u'In many villages what establishment could be called the focal point of the community?', u'id': u'56dede3c3277331400b4d785', u'answers': [{u'text': u'the pub', u'answer_start': 393}]},
                        {u'question': u"What is the term 'pub' short for?", u'id': u'56dfb4987aa994140058e003', u'answers': [{u'text': u'public house', u'answer_start': 16}]},
                        {u'question': u'Where in the United States are pubs located?', u'id': u'56dfb4987aa994140058e004', u'answers': [{u'text': u'New England', u'answer_start': 255}]},
                        {u'question': u'What continental European country has pubs?', u'id': u'56dfb4987aa994140058e005', u'answers': [{u'text': u'Denmark', u'answer_start': 243}]},
                        {u'question': u'Other than the United States, where in North America are pubs located?', u'id': u'56dfb4987aa994140058e006', u'answers': [{u'text': u'Canada', u'answer_start': 235}]},
                        {u'question': u'Who said that pubs are the heart of England?', u'id': u'56dfb4987aa994140058e007', u'answers': [{u'text': u'Samuel Pepys', u'answer_start': 371}]}]},
               {u'context': u'The history of pubs can be traced back to Roman taverns, through the Anglo-Saxon alehouse to the development of the modern tied house system in the 19th century.',
               u'qas': [{u'question': u'How far back does the history of pubs go back?', u'id': u'56dedf02c65bf219000b3d93', u'answers': [{u'text': u'to Roman taverns', u'answer_start': 39}]},
                        {u'question': u'What was the Anglo-Saxon pup called? ', u'id': u'56dedf02c65bf219000b3d94', u'answers': [{u'text': u'alehouse', u'answer_start': 81}]},
                        {u'question': u'What is a pub tied to in the 19th century?', u'id': u'56dedf02c65bf219000b3d95', u'answers': [{u'text': u'the modern tied house system', u'answer_start': 112}]},
                        {u'question': u'What Roman businesses were analogous to modern day pubs?', u'id': u'56dfb4d6231d4119001abc97', u'answers': [{u'text': u'taverns', u'answer_start': 48}]},
                        {u'question': u'What similar establishments existed in the Anglo-Saxon world?', u'id': u'56dfb4d6231d4119001abc98', u'answers': [{u'text': u'alehouse', u'answer_start': 81}]},
                        {u'question': u'In what century did the tied house system develop?', u'id': u'56dfb4d6231d4119001abc99', u'answers': [{u'text': u'19th century', u'answer_start': 148}]}]},
               {u'context': u'Historically, pubs have been socially and culturally distinct from caf\xe9s, bars and German beer halls. Most pubs offer a range of beers, wines, spirits, and soft drinks and snacks. Traditionally the windows of town pubs were of smoked or frosted glass to obscure the clientele from the street but from the 1990s onwards, there has been a move towards clear glass, in keeping with brighter interiors.',
               u'qas': [{u'question': u'Why were the windows of town pubs made of smoked or frosted glass traditionally?', u'id': u'56dee02f3277331400b4d79b', u'answers': [{u'text': u'to obscure the clientele from the street', u'answer_start': 251}]},
                        {u'question': u'What fares do most pubs offer?', u'id': u'56dee02f3277331400b4d79d', u'answers': [{u'text': u'beers, wines, spirits, and soft drinks and snacks', u'answer_start': 129}]},
                        {u'question': u'What are traditional pub windows made out of?', u'id': u'56dfb587231d4119001abca1', u'answers': [{u'text': u'smoked or frosted glass', u'answer_start': 227}]},
                        {u'question': u'What are the windows of 1990s and later pubs often made of?', u'id': u'56dfb587231d4119001abca2', u'answers': [{u'text': u'clear glass', u'answer_start': 350}]},
                        {u'question': u'Aside from beverages, what types of food do pubs typically offer?', u'id': u'56dfb587231d4119001abca3', u'answers': [{u'text': u'snacks', u'answer_start': 172}]}]},
"""

               
#               if question.find('who') != -1:
#                   for word,pos in pos_question:
#                       if pos == 'NNP':
#                           for sentence in sentences:
#                               if sentence.find(word) != -1:
#                                   sentence_pos = nltk.pos_tag(sentence)
#                                   sentence_iob = nltk.tree2conlltags(nltk.ne_chunk(sentence_pos)
#                                   noun = []
#                                   person = []
#                                   for word, pos_tag, iob in sentence_iob:
#                                       if iob == 'B-PERSON' or iob =='I-PERSON':
#                                           person.append(iob)
#                                       else:
#                                           if pos_tag == 'NNP':
#                                               noun.append(pos_tag)
#                                    if len(person) != 0:
#                                        ans_arr.append(person)
#                                    else:
#                                        ans_arr.append(noun)
