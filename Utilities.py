import os
import sys
import datetime
import pandas as pd
import math
import csv
from pycorenlp import StanfordCoreNLP
import nltk
from textblob import TextBlob
import re
import datetime
import re
import sys,os
import glob
from time import sleep

posts = nltk.corpus.nps_chat.xml_posts()[:100]
posts=list(posts)
questions_list = []
final_list = {}

nlp = StanfordCoreNLP('http://localhost:9000')


def removePunctuation(s):
    # Sample string
    return re.sub(r'[^a-zA-Z0-9]', '', s)

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def sentiment_analysis(sentence):
    text = sentence
    output = nlp.annotate(text, properties={
        'annotators': 'sentiment,parse',
        'outputFormat': 'json'
    })
    # print(output['sentences'][0]['sentiment'])
    return output['sentences'][0]['sentiment']


def subjectivty():
    featuresets = [(dialogue_act_features(post.text), post.get('class'))
                   for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    return classifier


def text_blob(sentence1):
    sentence = TextBlob(sentence1)
    if float(sentence.sentiment.subjectivity) > 0.2:
        if sentiment_analysis(sentence1) == 'Neutral':
            da = 'statement-non-opinion'
        else:
            da = 'statement-opinion' + sentiment_analysis(sentence1)

    else:
        da = 'statement-non-opinion'
    return da

#reading initial file names
def read_filename(inputSlot, inputFileName):
    rel_path = "/"
    abs_data_path = os.getcwd()

    df = pd.read_csv(abs_data_path + "/"+inputFileName)

    dict1 = {}
    dict2 = {}
    for x in range(inputSlot + 1):
        # dict1[x]='list_slot'+str(x)
        # dict2[x]='final_list'+str(x)

        dict1[x] = list()
        dict2[x] = list()

    for index, row in df.iterrows():

        if row['slot0'] not in dict1[0]:
            dict1[0].append(row['slot0'])
        if row['slot1'] not in dict1[1]:
            dict1[1].append(row['slot1'])
        if inputSlot > 2:
            if row['slot2'] not in dict1[2]:
                dict1[2].append(row['slot2'])
        if inputSlot > 3:
            if row['slot3'] not in dict1[3]:
                dict1[3].append(row['slot3'])
        if inputSlot > 4:
            if row['slot4'] not in dict1[4]:
                dict1[4].append(row['slot4'])
    return dict1
#calculate

def calculate(list_initial_prompts):

        f1={}
        result={}

        for i in list_initial_prompts:

            classifier = subjectivty()
            da = ''
            if '?' in i:
                # print(i)
                # DA=classifier.classify(dialogue_act_features(i))
                # print( "1",i,classifier.classify(dialogue_act_features(i)))
                # da=classifier.classify(dialogue_act_features(i))
                questions_list.append(i)
            else:
                da = text_blob(i)
                f1[i] = da

        final_list.update(f1)
        result[0]=final_list
        result[1]=questions_list
        return result


# writing questions_list for prediction
def write_questions_for_predictions(ql):

    questions_list=ql
    rel_path = "\da-classification/data/test_statements2"
    file_name = os.getcwd() + rel_path+"/"+"test1.csv"

    with open(file_name, 'w', newline='') as f:
        myFields = ['text', 'act_tag']
        writer = csv.DictWriter(f, fieldnames=myFields)

        writer.writeheader()

        for key in ql:
            writer.writerow({'text': key, 'act_tag': 'qw'})

def read_predicted_questions():
    q_3 = {}
    rel_path = r"/predictions"
    abs_data_path = os.getcwd()+rel_path+"/predictions.csv"

    df = pd.read_csv(abs_data_path)

    for index, row in df.iterrows():
        if row['prediction'] == 'qw':
            q_3[row['DA']] = 'whQuestion'
        elif row['prediction'] == 'qo':
            q_3[row['DA']] = 'openQuestion'
        else:
            q_3[row['DA']] = 'ynQuestion'
    return q_3

    #final_list3 = dict(list(final_list3.items()) + list(q_3.items()))

def merge_list(dict1):
    dict2 = {}
    for key, index in dict1.items():
        print(key)
        key = removePunctuation(key)
        dict2[key]=index
    return dict2

def write_DA(inputfile,inputSlots,dict2):

    #abs_data_path = os.path.join(os.getcwd(), "/")
    rel_path="../.."
    abs_data_path=os.getcwd()+rel_path
    os.chdir(abs_data_path)

    with open('output_with_DAs.csv', 'w', newline='') as f:


        if inputSlots ==2:
            myFields = ['conversation_id', 'slot0', 'slot1', 'Dialog_act_slot0',
                            'Dialog_act_slot1']
            writer = csv.DictWriter(f, fieldnames=myFields)

            writer.writeheader()
            r1 = csv.reader(open(inputfile))
            for values in r1:
                a = ''
                b = ''

                if removePunctuation(values[1]) in dict2[0]:
                    a = dict2[0][removePunctuation(values[1])]

                if removePunctuation(values[2]) in dict2[1]:
                    b = dict2[1][removePunctuation(values[2])]


                writer.writerow(
                    {'conversation_id': values[0], 'slot0': values[1], 'slot1': values[2],
                     'Dialog_act_slot0': a, 'Dialog_act_slot1': b,
                     })
        elif inputSlots == 3:


            myFields = ['conversation_id', 'slot0', 'slot1','slot2', 'Dialog_act_slot0',
                            'Dialog_act_slot1','Dialog_act_slot2']
            writer = csv.DictWriter(f, fieldnames=myFields)

            writer.writeheader()
            r1 = csv.reader(open(inputfile))
            for values in r1:

                a = ''
                b = ''
                c = ''

                if removePunctuation(values[1]) in dict2[0]:
                    a = dict2[0][removePunctuation(values[1])]

                if removePunctuation(values[2]) in dict2[1]:
                    b = dict2[1][removePunctuation(values[2])]

                if removePunctuation(values[3]) in dict2[2]:
                    c = dict2[2][removePunctuation(values[3])]

                writer.writerow(
                    {'conversation_id': values[0], 'slot0': values[1], 'slot1': values[2],'slot2': values[3],
                     'Dialog_act_slot0': a, 'Dialog_act_slot1': b,'Dialog_act_slot2': c,
                     })

        elif inputSlots == 4:
            myFields = ['conversation_id', 'slot0', 'slot1','slot2','slot3', 'Dialog_act_slot0',
                            'Dialog_act_slot1','Dialog_act_slot2','Dialog_act_slot3']
            writer = csv.DictWriter(f, fieldnames=myFields)

            writer.writeheader()
            r1 = csv.reader(open(inputfile))
            for values in r1:
                a = ''
                b = ''
                c = ''
                d= ''
                if removePunctuation(values[1]) in dict2[0]:
                    a = dict2[0][removePunctuation(values[1])]

                if removePunctuation(values[2]) in dict2[1]:
                    b = dict2[1][removePunctuation(values[2])]

                if removePunctuation(values[3]) in dict2[2]:
                    c = dict2[2][removePunctuation(values[3])]

                if removePunctuation(values[4]) in dict2[3]:
                    d = dict2[3][removePunctuation(values[4])]

                writer.writerow(
                {'conversation_id': values[0], 'slot0': values[1], 'slot1': values[2], 'slot2': values[3],'slot3': values[4],
                     'Dialog_act_slot0': a, 'Dialog_act_slot1': b, 'Dialog_act_slot2': c,'Dialog_act_slot3': d,
                     })

        else:
            myFields = ['conversation_id', 'slot0', 'slot1','slot2','slot3','slot4', 'Dialog_act_slot0',
                            'Dialog_act_slot1','Dialog_act_slot2','Dialog_act_slot3','Dialog_act_slot4']
            writer = csv.DictWriter(f, fieldnames=myFields)

            writer.writeheader()
            r1 = csv.reader(open(inputfile))
            for values in r1:
                a = ''
                b = ''
                c = ''
                d = ''
                e=''
                if removePunctuation(values[1]) in dict2[0]:
                    a = dict2[0][removePunctuation(values[1])]

                if removePunctuation(values[2]) in dict2[1]:
                    b = dict2[1][removePunctuation(values[2])]

                if removePunctuation(values[3]) in dict2[2]:
                    c = dict2[2][removePunctuation(values[3])]

                if removePunctuation(values[4]) in dict2[3]:
                    d = dict2[3][removePunctuation(values[4])]

                if removePunctuation(values[5]) in dict2[4]:
                    e = dict2[4][removePunctuation(values[5])]

                writer.writerow(
                        {'conversation_id': values[0], 'slot0': values[1], 'slot1': values[2], 'slot2': values[3],
                     'slot3': values[4],'slot4': values[5],
                     'Dialog_act_slot0': a, 'Dialog_act_slot1': b, 'Dialog_act_slot2': c, 'Dialog_act_slot3': d,'Dialog_act_slot4': e,
                     })

