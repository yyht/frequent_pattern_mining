import data_processor
import numpy as np

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"stop_word_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"ouput_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"n_gram", None,
	"Input TF example files (can be a glob or comma separated).")

def main(_):
	with open(FLAGS.train_file, "r") as frobj:
		corpus_lst = []
		for line in frobj:
			corpus_lst.append(line.strip())

	stop_word_lst = []
	with open(FLAGS.stop_word_file, "r") as frobj:
		for line in frobj:
			stop_word_lst.append(line.strip())

	LABEL_SPLITTER = "__label__"
	re_pattern = "({}{})".format(LABEL_SPLITTER, "\d.")
	corpus, label = data_processor.read_corpus(corpus_lst, 
											   stop_word_lst, 
											   re_pattern, 
											   LABEL_SPLITTER,
											  if_debug=False)

	label_corpus_dict = {}
	for lab, cor in zip(label, corpus):
		if lab in label_corpus_dict:
			label_corpus_dict[lab].append(cor)
		else:
			label_corpus_dict[lab] = [cor]
	for key in label_corpus_dict:
		print(len(label_corpus_dict[key]), "===", key)

	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(ngram_range=(1, FLAGS.n_gram), 
						max_df=1.0, min_df=1, 
						analyzer="word", smooth_idf=False)

	label_ngram_dict = {}

	import _pickle as pkl

	for key in label_corpus_dict:
		X = vectorizer.fit_transform(label_corpus_dict[key])
		idf = vectorizer.idf_
		feature_names = vectorizer.get_feature_names()
		ngram_idf = list(zip(feature_names, idf))
		ngram_sorted = sorted(ngram_idf, key=lambda item:item[1], reverse=True)
		label_ngram_dict[key] = ngram_sorted

	pkl.dump(label_ngram_dict, open(FLAGS.ouput_file, "wb"))


if __name__ == "__main__":
	tf.app.run()