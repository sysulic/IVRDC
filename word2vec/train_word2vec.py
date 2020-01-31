import os
import gensim
import numpy as np

## Training Word2Vector Model
class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for line in open(os.path.join(self.dirname)):
			line.strip()
			yield line.split()

print 'Training model...'
sentences = MySentences('VG_corpus/VG_corpus.txt')
# skip-gram: sg=1, CBOW:sg=0
model = gensim.models.Word2Vec(sentences, sg=1, size=300, window=2, \
	min_count=5, hs=0, workers=4)

print 'Saving model...'
model.save('300d_word2vec_vg')

# print 'Loading model...'
# model = gensim.models.Word2Vec.load('word2vec_model')

# with open('../dataset/vrd/object_class.txt', 'r') as file:
# 	obj_lines = file.readlines()
# 	obj_line_num = len(obj_lines)
# file.close()

# print 'Writing vector into file...'
# for word in obj_lines:
# 	word = word.strip().replace(' ', '-')
# 	np.savez('vectors.npz', model[word])
