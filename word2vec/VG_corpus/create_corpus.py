import re
import nltk
import json as js
import unicodedata
import enchant
from nltk.metrics import edit_distance
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

en_dict = enchant.Dict("en_US")

def replace(word):
    if word=='wearing':
        return 'wear'

    if en_dict.check(word):
        return word
    suggestions = en_dict.suggest(word)
    if len(suggestions) and isinstance(suggestions[0],unicode):
        return word
    else:
        if suggestions and edit_distance(word, suggestions[0]) <= 2:
            return suggestions[0]
    return word
    
def lemmatize_sentence(sentence, pos_tag):
    res=[]
    lemmatizer = WordNetLemmatizer()
    seq = re.split(r' |/', sentence)
    for word in seq:
        word = word.lower()
        if word==u'trees' or word==u'glasses' or word==u'has' or word==u'pants'\
        or word==u'shorts' or word==u'jeans' or word==u'sunglasses' or word==u'shoes' or word==u'lying':
            res.append(word)
        else:
            res.append(lemmatizer.lemmatize(word, pos=pos_tag))
    return res

def write_name(object_, file):
	# Lemmatization 
	normalized_obj_list = lemmatize_sentence(object_, 'n')  
	normalized_str = ''

	length = len(normalized_obj_list)
	# print normalized_obj_list
	for k in xrange(length):
		# print normalized_obj[k],
		tmp_obj = unicodedata.normalize('NFKD', normalized_obj_list[k]).encode('ascii', 'ignore').lower()
		tmp_obj = tmp_obj.strip('\"/!.`\'[]?()\x00')

		if tmp_obj=='.' or tmp_obj==',' or tmp_obj== '\'s' or tmp_obj== '('\
			or tmp_obj=='\'' or tmp_obj=='`' or tmp_obj=='\'\'' or tmp_obj=='``'\
			or tmp_obj=='[' or tmp_obj==']' or tmp_obj == '?' or tmp_obj== ')'\
			or tmp_obj=='a' or tmp_obj=='his' or tmp_obj=='her' or tmp_obj=='their' or tmp_obj=='that'\
			or tmp_obj=='the' or tmp_obj=='-' or tmp_obj==' ' or tmp_obj=='' or tmp_obj==None:
			continue 
            
		tmp_obj = replace(tmp_obj)

		if k!= length-1 and tmp_obj=='traffic' and normalized_obj_list==u'light':
			normalized_str += tmp_obj + '-'
		else:
			if k == length-1:
				normalized_str += tmp_obj
			else:
				normalized_str += tmp_obj + ' '     
	# normalized_str = normalized_str.replace(' ', '-')

	file.write(normalized_str + ' ')

def write_predicate(predicate, file):
	# Lemmatization
	normalized_str = ''
	normalized_pre_list = lemmatize_sentence(predicate, 'v')
	# print normalized_pre_list
	length = len(normalized_pre_list)
	for k in xrange(length):
		# print normalized_pre_list[k], type(normalized_pre_list[k])
		tmp_pre = unicodedata.normalize('NFKD', normalized_pre_list[k]).encode('ascii', 'ignore').lower()
		tmp_pre = tmp_pre.strip('\"/!.`\'')
		if tmp_pre == 'a' or tmp_pre == 'the' or tmp_pre == 'be' or tmp_pre == 'that'\
			or tmp_pre == 'her' or tmp_pre == 'his' or tmp_pre == 'their' or tmp_pre == 'this'\
			or tmp_pre == ' ' or tmp_pre == '' or tmp_pre == 'an':
			continue

		tmp_pre = replace(tmp_pre)

		if k == length-1 or (k == length-2 and (normalized_pre_list[k+1] == 'a' or normalized_pre_list[k+1] == 'the'\
			or normalized_pre_list[k+1] == 'be' or normalized_pre_list[k+1] == 'that' or normalized_pre_list[k+1] == 'an'\
			or normalized_pre_list[k+1] == 'her' or normalized_pre_list[k+1] == 'his' or normalized_pre_list[k+1] == 'their')):
			normalized_str += tmp_pre
		else:
			normalized_str += tmp_pre + ' '
	normalized_str = normalized_str.replace(' ', '-')
	file.write(normalized_str + ' ')

with open("../../dataset/vg/relationships.json", 'r') as file:
	vg = js.load(file)
file.close()

file = open("vg_corpus.txt", 'w+')
num_vg = len(vg)

for i in xrange(num_vg):
	data = vg[i]
	relationships = data[u'relationships']
	num_rs = len(relationships)
	print(i)
	for j in xrange(num_rs):
		rs = relationships[j]

		predicate = rs[u'predicate'].strip()
        
		# If the relationship has predicate
		if predicate=='':
			continue
		else:  
			# subject #
			sub = rs[u'subject']
			if sub.has_key(u'name'):
				sub_name = sub[u'name'].strip()
				# print sub_name
				write_name(sub_name, file)
			else:
				names = sub[u'names']
				for k in xrange(len(names)):
					sub_name = names[k].strip()
					write_name(sub_name, file)

			# predicate #
			write_predicate(predicate, file)
			# print length,

			obj = rs[u'object']
			if obj.has_key(u'name'):
				obj_name = obj[u'name'].strip()
				write_name(obj_name, file)
			else:
				names = obj[u'names']
				for k in xrange(len(names)):
					obj_name = names[k].strip()
					write_name(obj_name, file)
			file.write('\n')

# data = vg[49984]
# relationships = data[u'relationships']
# num_rs = len(relationships)

# for j in xrange(num_rs):
# 	rs = relationships[j]

# 	predicate = rs[u'predicate'].strip()
        
# 	# If the relationship has predicate
# 	if predicate=='':
# 		continue
# 	else:  
# 		# subject #
# 		sub = rs[u'subject']
# 		if sub.has_key(u'name'):
# 			sub_name = sub[u'name'].strip()
# 			# print sub_name
# 			write_name(sub_name, file)
# 		else:
# 			names = sub[u'names']
# 			for k in xrange(len(names)):
# 				sub_name = names[k].strip()
# 				write_name(sub_name, file)
# 			# predicate #
# 		write_predicate(predicate, file)
# 		# print length,
# 		obj = rs[u'object']
# 		if obj.has_key(u'name'):
# 			obj_name = obj[u'name'].strip()
# 			write_name(obj_name, file)
# 		else:
# 			names = obj[u'names']
# 			for k in xrange(len(names)):
# 				obj_name = names[k].strip()
# 				write_name(obj_name, file)
# 		file.write('\n')
       
file.close()

