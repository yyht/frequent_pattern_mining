import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import re
from hanziconv import HanziConv
import pandas as pd

import random


def full2half(s):
	n = []
	for char in s:
		num = ord(char)
		if num == 0x3000:
			num = 32
		elif 0xFF01 <= num <= 0xFF5E:
			num -= 0xfee0
		num = chr(num)
		n.append(num)
	return ''.join(n)

def load_stop_words(stop_word_path):
	with codecs.open(stop_word_path, "r", "utf-8") as frobj:
		word_lst = []
		for line in frobj:
			word = line.strip()
			word_lst.append(word)
	return word_lst

def remove_stop_word(text, stop_word_lst):
	text = text.split()
	text_ = []
	for word in text:
		if word in stop_word_lst:
			continue
		else:
			text_.append(word)
	return " ".join(text_)

def clean(text):
	text = text.strip()
	text = HanziConv.toSimplified(text)
	text = full2half(text)
	text = re.sub("\\#.*?#|\\|.*?\\||\\[.*?]", "", text)
	# text = re.sub("\s*", "", text)
	return text

def extractor(content, re_pattern, LABEL_SPLITTER="__label__"):
	element_list = re.split(re_pattern, content)
	text_a = clean(element_list[-1])
	input_labels = clean(element_list[1]).split(LABEL_SPLITTER)[-1]
	return text_a, input_labels

def read_corpus(corpus_lst, stop_word_lst, re_pattern, LABEL_SPLITTER, if_debug):
	content_lst = []
	label_lst = []

	if if_debug:
		end_index = 10000
	else:
		end_index = -1

	random.shuffle(corpus_lst)

	for content in corpus_lst[0:end_index]:
		text, label = extractor(content, re_pattern, LABEL_SPLITTER)
		text = remove_stop_word(text, stop_word_lst)
		if len(text) >= 2:
			content_lst.append(text)
			label_lst.append(label)
	return content_lst, label_lst

def write_to_csv(csv_path, content_lst, label_lst):
	data_dict = {
		"corpus":corpus_lst,
		"label":label_lst
	}

	df = pd.DataFrame(data_dict)
	pd.to_csv(csv_path, data_dict)






