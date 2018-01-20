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

#sort model descending order and return the latest
def fetch_model():
    print("in here")
    rel_path = "\model"
    abs_data_path = os.getcwd()+rel_path
    os.chdir(abs_data_path)
    print(abs_data_path)
    files = glob.glob("*.hdf5")
    files.sort(key=os.path.getmtime)
    rel_path = "../.."
    abs_data_path = os.getcwd() + rel_path
    os.chdir(abs_data_path)
    return files[-1]

# execute all question classification
def questions_predictions():

    rel_path = "da-classification"
    abs_data_path = os.path.join(os.getcwd(), rel_path)

    os.chdir(abs_data_path)
    os.system("python main.py --train data/train_statements2 data/test_statements2")
    print(abs_data_path)

    os.system("python main.py --predict data/train_statements2 data/test_statements2 model/"+ fetch_model())


    rel_path = "predictions"
    abs_data_path = os.path.join(os.getcwd(), rel_path)
