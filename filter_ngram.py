import numpy as np
from collections import OrderedDict

def filter_ngram(ngram_dict, source_key, target_key):
	src_ngram_dict = OrderedDict(ngram_dict[source_key])
	tgt_ngram_dict = OrderedDict(ngram_dict[target_key])

	tgt_original_dict = OrderedDict({})

	for key in tgt_ngram_dict:
		if key in src_ngram_dict:
			continue
		else:
			tgt_original_dict[key] = tgt_ngram_dict[key]
	return tgt_original_dict