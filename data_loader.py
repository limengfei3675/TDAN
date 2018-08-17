import json
import pdb

import h5py
import numpy as np
import os
from os.path import join
import pickle

class Data_loader:
	# Before using data loader, make sure your data/ folder contains required files
	def __init__(self):
		print("Initialize Data_laoder")
	def right_align(self, seq, lengths):
		v = np.zeros(np.shape(seq))
		ss = seq.shape
		N = ss[1]
		for i in range(ss[0]):
			v[i][N - lengths[i]:N - 1] = seq[i][0:lengths[i] - 1]
		return v

	def get_qa_data(self, data_dir, data_set = "train"):
		dataset = {}
		# load json file
		print('loading json file...')
		with open(join(data_dir,"3000_data_prepro.json"),"r") as data_file:
			data = json.load(data_file)
		for key in data.keys():
			dataset[key] = data[key]

		train_test_data = np.load(join(data_dir, "v2_3000_qa_data_train_test.npy")).item()
		qa_data = train_test_data[data_set]
		qa_data["images_pos"] = qa_data["images_pos"] - 1
		qa_data['questions'] = self.right_align(qa_data['questions'], qa_data['questions_length'])
		if data_set == "train":
			qa_data["answers"] = qa_data["answers"] - 1
		print("Questions : %d" %(len(qa_data["questions"])))
		return dataset, qa_data

	def get_test_dev_qa_data(self, data_dir):
		dataset = {}
		# load json file
		print('loading json file...')
		with open(join(data_dir,"3000_data_prepro.json"),"r") as data_file:
			data = json.load(data_file)
		for key in data.keys():
			dataset[key] = data[key]
		# with open(join(data_dir, "3000_test_dev_qa_data.npy",),"rb") as f:
		#	 pickle_data = pickle.load(f)
			# keys ['questions_id', 'questions', 'answers', 'images_pos', 'questions_length', 'images_id']
		qa_data = np.load(join(data_dir, "v2_3000_qa_data_test_dev.npy")).item()
		qa_data["images_pos"] = qa_data["images_pos"] - 1
		qa_data['questions'] = self.right_align(qa_data['questions'], qa_data['questions_length'])

		print("Questions : %d" %(len(qa_data["questions"])))
		return dataset, qa_data

	def get_image_feature(self, data_dir="/disk5", train = True):
		if train:
			data = np.load(join(data_dir, "coco_features_train_val.npy")).item()
		else:
			data = np.load(join(data_dir, "coco_features_test.npy")).item()
		return data

	def get_next_batch(self,batch_no, batch_size, max_question_length, qa_data, image_features, train=True, questions_feats=None):
		train_length = len(qa_data["questions"])
		si = (batch_no * batch_size) % train_length
		# unique_img_train[img_pos_train[0]]
		ei = min(train_length, si + batch_size)
		n = ei - si
		pad_n = n
		n = batch_size
		answer = np.zeros(n)
		image_feature = np.ndarray((n, 36, 2048))
		appendixes = np.zeros((n,max_question_length,max_question_length), dtype = 'int32')
		# ---------------
		if questions_feats is None:
			question = np.ndarray((n, max_question_length), dtype=np.int32)
		else:
			question = np.ndarray((n, 4800), dtype=np.float32)
		count = 0
		# pdb.set_trace()
		questions_id = []
		images_id = []
		for i in range(si, ei):
			answer[count] = int(qa_data["answers"][i])
			img_id = int(qa_data["images_id"][i])
			# pdb.set_trace()
			image_feature[count, :] = image_features[img_id]
			if questions_feats is None:
				question[count, :] = qa_data["questions"][i]
			else:
				question[count, :] = questions_feats[qa_data["questions_id"][i]]
			questions_id.append(int(qa_data["questions_id"][i]))
			images_id.append(img_id)

			zero_length = 0
			for step in range(max_question_length):
				if question[count][step] != 0:
					zero_length = step
					break
			for j in range(zero_length,max_question_length):
				appendixes[count][j][j-zero_length] = 1
			count += 1

		for i in range(n - pad_n):
			r_num = np.random.randint(0, train_length - 1)
			answer[count] = int(qa_data["answers"][r_num])
			img_id = int(qa_data["images_id"][r_num])
			image_feature[count, :] = image_features[img_id]
			if questions_feats is None:
				question[count, :] = qa_data["questions"][r_num]
			else:
				question[count, :] = questions_feats[qa_data["questions_id"][r_num]]
			questions_id.append(int(qa_data["questions_id"][r_num]))
			images_id.append(img_id)

			zero_length = 0
			for step in range(max_question_length):
				if question[count][step] != 0:
					zero_length = step
					break
			for j in range(zero_length,max_question_length):
				appendixes[count][j][j-zero_length] = 1

			count += 1
		if train:
			return image_feature, question, answer,appendixes,images_id, questions_id
		else:
			return image_feature, question, answer,appendixes,images_id, questions_id

	def get_next_batch_test(self,batch_no, batch_size, max_question_length, qa_data, image_features):
		train_length = len(qa_data["questions"])
		si = (batch_no * batch_size) % train_length
		# unique_img_train[img_pos_train[0]]
		ei = min(train_length, si + batch_size)
		n = ei - si
		pad_n = n
		n = batch_size
		image_feature = np.ndarray((n, 36, 2048))
		appendixes = np.zeros((n,max_question_length,max_question_length), dtype = 'int32')
		header = np.zeros(n)
		# ---------------
		question = np.ndarray((n, max_question_length), dtype=np.int32)
		count = 0
		# pdb.set_trace()
		questions_id = []
		images_id = []
		for i in range(si, ei):
			img_id = int(qa_data["images_id"][i])
			image_feature[count, :] = image_features[img_id]
			question[count, :] = qa_data["questions"][i]
			questions_id.append(qa_data["questions_id"][i])
			images_id.append(qa_data["images_id"][i])
			zero_length = 0
			for step in range(max_question_length):
				if question[count][step] != 0:
					zero_length = step
					break
			header[count] = question[count][zero_length]
			for j in range(zero_length,max_question_length):
				appendixes[count][j][j-zero_length] = 1
			count += 1

		for i in range(n - pad_n):
			r_num = np.random.randint(0, train_length - 1)
			img_id = int(qa_data["images_id"][r_num])
			image_feature[count, :] = image_features[img_id]
			question[count, :] = qa_data["questions"][r_num]
			questions_id.append(qa_data["questions_id"][r_num])
			images_id.append(qa_data["images_id"][r_num])
			zero_length = 0
			for step in range(max_question_length):
				if question[count][step] != 0:
					zero_length = step
					break
			header[count] = question[count][zero_length]
			for j in range(zero_length,max_question_length):
				appendixes[count][j][j-zero_length] = 1
			count += 1
		return image_feature, question, answer,appendixes,images_id, questions_id

	def get_pre_embedding(self, data_dir="./data"):
		word_embed = np.load(join(data_dir, "v2_3000_train_val_glove_embedding.npy"))
		return word_embed

	def get_resnet_rcnn_next_batch(self, batch_no, batch_size, max_question_length, qa_data, image_features,
								   train=True,
								   id_to_index=None,
								   questions_feats=None,
								   image_id_to_path=None):
		train_length = len(qa_data["questions"])
		si = (batch_no * batch_size) % train_length
		# unique_img_train[img_pos_train[0]]
		ei = min(train_length, si + batch_size)
		n = ei - si
		pad_n = n
		n = batch_size
		answer = np.zeros((n))
		resnet_image_feature = np.ndarray((n, 14, 14, 2048))
		rcnn_image_feature = np.ndarray((n, 36, 2048))
		# ---------------
		if questions_feats is None:
			question = np.ndarray((n, max_question_length), dtype=np.int32)
		else:
			question = np.ndarray((n, 4800), dtype=np.float32)
		count = 0
		images_id = []
		questions_id = []
		# pdb.set_trace()
		for i in range(si, ei):
			answer[count] = qa_data["answers"][i]
			img_id = qa_data["images_id"][i]
			if train:
				feat_path = "/home/hbliu/data/coco_features/resnet-152/res5c/"+image_id_to_path[str(img_id)]
			else:
				test_image_name = 'test2015/COCO_test2015_%.12d.npy' % (img_id)
				feat_path = "/home/hbliu/data/coco_features/resnet-152/res5c/"+test_image_name
			image_feat = np.load(feat_path)
			resnet_image_feature[count, :] = np.float32(image_feat)
			rcnn_image_feature[count, :] = image_features[img_id]
			if questions_feats is None:
				question[count, :] = qa_data["questions"][i]
			else:
				q_id = str(qa_data["questions_id"][i])
				q_index = id_to_index[q_id]
				question[count, :] = questions_feats[q_index]
			images_id.append(img_id)
			questions_id.append(qa_data["questions_id"][i])
			count += 1

		for i in range(n - pad_n):
			r_num = np.random.randint(0, train_length-1)
			answer[count] = qa_data["answers"][r_num]
			img_id = qa_data["images_id"][r_num]
			if train:
				feat_path = "/home/hbliu/data/coco_features/resnet-152/res5c/" + image_id_to_path[str(img_id)]
			else:
				test_image_name = 'test2015/COCO_test2015_%.12d.npy' % (img_id)
				feat_path = "/home/hbliu/data/coco_features/resnet-152/res5c/" + test_image_name
			# feat_path = feat_path.replace(".jpg", ".npy")
			image_feat = np.load(feat_path)
			resnet_image_feature[count, :] = np.float32(image_feat)
			rcnn_image_feature[count, :] = image_features[img_id]
			if questions_feats is None:
				question[count, :] = qa_data["questions"][r_num]
			else:
				q_id = str(qa_data["questions_id"][r_num])
				q_index = id_to_index[q_id]
				question[count, :] = questions_feats[q_index]
			images_id.append(img_id)
			questions_id.append(qa_data["questions_id"][r_num])
			count += 1
		if train:
			return resnet_image_feature, rcnn_image_feature, question, answer
		else:
			return resnet_image_feature, rcnn_image_feature, question, answer, images_id, questions_id

	# def get_skip_thought_feature(self):