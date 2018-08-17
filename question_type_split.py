import tensorflow as tf, numpy as np
import os
import time
import nltk
import numpy as np
import tensorflow as tf
from os.path import join
import argparse
import data_loader
import dual_cross_guided_att_vqamodel
import json
import h5py
import pdb
import sys
sys.path.insert(0, "../")

def do_business(args):
	loader = data_loader.Data_loader()
	
	# pdb.set_trace()
	print("Loading data...............")
	dataset, train_data = loader.get_qa_data(args.data_dir, data_set="train")
	
	print("Done !")
	# print("Image data length is %d " %len(image_features))
	word_num = len(dataset["ix_to_word"])
	print("Vocab size is %d "%word_num)
	print("Answer num is %d "%len(dataset["ix_to_ans"]))
	
	train_question_str = json.load(open('/home/lmf/mcan_for_vqa2/data/vqa_v2_question_train_train_val.json', 'r'))
	
	train_length = len(train_data["questions"])
	header_and_answers_map = {}
	# question_0s = {}
	question_0s = []
	max_question_length = args.questoin_max_length
	qa_data = train_data
	
	for i in range(train_length):
	
		questions_id = qa_data["questions_id"][i]
		# question_str.append(train_question_str[str(questions_id)].encode())
		question_str = train_question_str[str(questions_id)]
		question_str_split = question_str.split(' ')
		# for question_str_0 in question_str_split:
		question_str_0 = question_str_split[0]# + ' ' + question_str_split[1]
		if not (question_str_0 in question_0s):
			# question_0s[question_str_0] = 1
			question_0s.append(question_str_0)
		
		# else:
			# question_0s[question_str_0] += 1
		"""
		zero_length = 0
		for step in range(max_question_length):
			if qa_data["questions"][i][step] != 0:
				zero_length = step
				break
		pdb.set_trace()
		header = qa_data["questions"][i][zero_length] #+ qa_data["questions"][i][zero_length + 1]
		answer = qa_data["answers"][i]
		if header not in header_and_answers_map.keys():
			l=[]
			l.append(answer)  
			header_and_answers_map[header]=l
		else:
			if answer not in header_and_answers_map[header]:
				header_and_answers_map[header].append(answer)
		"""
	"""
	with open(join(args.data_dir,"question_split.json"), "w") as f:
		f.write(str(header_and_answers_map))
		f.close()
	
	with open(join(args.data_dir,"question_split.json"), "r") as f:
		a = f.read()
		header_and_answers_map = eval(a)
		f.close()
	"""
	# sorted_x = sorted(question_0s.iteritems(), key= lambda d:d[1], reverse=True)
	aaa = nltk.pos_tag(question_0s)
	pdb.set_trace()
	with open(join(args.data_dir,"question_split.json"), "w") as f:
		f.write(str(question_0s))
		f.close()
	
	
	

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=100,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.0003,
						help='Batch Size')
	parser.add_argument('--epochs', type=int, default=1000,
						help='Expochs')
	parser.add_argument('--input_embedding_size', type=int, default=300,
						help='word embedding size')
	parser.add_argument('--num_answers', type=int, default=3000,
						help='output answers nums')
	parser.add_argument('--questoin_max_length', type=int, default=20,
						help='output answers nums')
	parser.add_argument('--data_dir', type=str, default="data",
						help='output answers nums')
	parser.add_argument('--save_path', type=str, default="./save/dual_cross_guided_att_vqamodel",
						help='save path')
	args = parser.parse_args()
	print(args)
	do_business(args=args)

if __name__ == '__main__':
	main()