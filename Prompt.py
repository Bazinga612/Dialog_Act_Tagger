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
import Utilities as ut
import Questions_classifier as qc

# user_entered input
my_questions_list = []
print('Please enter the number of slots present')
inputSlot = int(input())
print('Please enter the FileName present')
inputFileName = input()
d=ut.read_filename(inputSlot, inputFileName)



dict2={}
for i in range(inputSlot):
    dict2[i] = ut.calculate(d[i])
ql=[]

for i in range(len(dict2[i])):
    ql.extend(dict2[i][1])

ut.write_questions_for_predictions(ql)

qc.questions_predictions()
predicted_questions=ut.read_predicted_questions()

dict3={}
dict4={}
for i in range(inputSlot):
    dict3[i] = dict(list(dict2[i][0].items()) + list(predicted_questions.items()))
    dict4[i]=ut.merge_list(dict3[i])
ut.write_DA(inputFileName,inputSlot,dict4)

