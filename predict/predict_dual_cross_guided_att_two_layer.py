import json
import pdb

import h5py
import numpy as np
import tensorflow as tf

import argparse

import sys
sys.path.insert(0, "../")
import data_loader
import dual_cross_guided_att_vqamodel_2


def get_next_batch(batch_no, batch_size, max_question_length, qa_data, image_features):
	train_length = len(qa_data["questions"])
	si = (batch_no * batch_size) % train_length
	# unique_img_train[img_pos_train[0]]
	ei = min(train_length, si + batch_size)
	n = ei - si
	pad_n = n
	n = batch_size
	image_feature = np.ndarray((n, 36, 2048))
	appendixes = np.zeros((n,max_question_length,max_question_length), dtype = 'int32')
	# ---------------
	question = np.ndarray((n, max_question_length), dtype=np.int32)
	count = 0
	# pdb.set_trace()
	questions_id = []
	images_id = []
	for i in range(si, ei):
		img_id = qa_data["images_id"][i]
		image_feature[count, :] = image_features[img_id]
		question[count, :] = qa_data["questions"][i]
		questions_id.append(qa_data["questions_id"][i])
		images_id.append(qa_data["images_id"][i])
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
		img_id = qa_data["images_id"][r_num]
		image_feature[count, :] = image_features[img_id]
		question[count, :] = qa_data["questions"][r_num]
		questions_id.append(qa_data["questions_id"][r_num])
		images_id.append(qa_data["images_id"][r_num])
		zero_length = 0
		for step in range(max_question_length):
			if question[count][step] != 0:
				zero_length = step
				break
		for j in range(zero_length,max_question_length):
			appendixes[count][j][j-zero_length] = 1
		count += 1
	return image_feature, question, images_id,appendixes, questions_id

def test(args):
	loader = data_loader.Data_loader()
	
	word_embed = loader.get_pre_embedding(args.data_dir)
	word_embed = np.float32(word_embed)
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	# pdb.set_trace()
	# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
	# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

	model = dual_cross_guided_att_vqamodel_2.Answer_Generator(
		batch_size=args.batch_size,
		dim_image=[6, 6, 2048],
		dim_hidden=512,
		max_words_q=args.questoin_max_length,
		drop_out_rate_0=1,
		drop_out_rate_1=1,
		drop_out_rate=1,
		num_output=args.num_answers,
		pre_word_embedding=word_embed)
	test_image, test_question, test_answer_prob,test_appendixes,test_vis_prob2,test_ques_prob2,test_model_att,test_image_att_mfh_1,test_image_att_mfh_2 = model.solver()
	
	sess.run(tf.global_variables_initializer())
	# Logging
	saver = tf.train.Saver()
	curr_epoch = 103
	model_file = '/disk5/limf/save/dual_cross_guided_att_vqamodel/model-' + str(curr_epoch)
	print("Restore models ...\n", model_file)
	saver.restore(sess, model_file)
	
	# pdb.set_trace()
	print("Loading data...............")
	dataset, test_data = loader.get_qa_data(args.data_dir, data_set="test")
	# dataset, test_data = loader.get_test_dev_qa_data(args.data_dir)
	# pdb.set_trace()
	print("Loading image features ...")
	image_features = loader.get_image_feature(train=False)
	print("Image feat length ", len(image_features))
	print("Done !")
	# print("Image data length is %d " %len(image_features))
	word_num = len(dataset["ix_to_word"])
	print("Vocab size is %d " % word_num)
	print("Answer num is %d " % len(dataset["ix_to_ans"]))
	print("Loading pre-word-embedding ......")
	
	# pdb.set_trace()
	for itr in range(1):
		batch_no = 0
		# pdb.set_trace()
		open_result = []
		mc_result = []
		test_length = len(test_data["questions"])
		print("length ---------------",test_length)
		length = 1
		vis2 = []
		ques2 = []
		model =[]
		vis_mfh_1 = []
		vis_mfh_2 = []
		images_id = []
		questions_id = []
		bbox = []
		boxes = np.load("/home/lmf/MCAN/data/test_image_box_top_36.npy").item()
		while (batch_no * args.batch_size) < test_length:
			# image_feature, question, answer,images_id, questions_id
			curr_image_feat, curr_question, curr_image_id,curr_appendixes, curr_ques_id,header = loader.get_next_batch_test(
				batch_no=batch_no,
				batch_size=args.batch_size,
				max_question_length=args.questoin_max_length,
				qa_data=test_data,
				image_features=image_features)
			# pdb.set_trace()
			answer_prob,vis_prob2,ques_prob2,model_att,image_att_mfh_1,image_att_mfh_2 = sess.run([test_answer_prob,test_vis_prob2,test_ques_prob2,test_model_att,test_image_att_mfh_1,test_image_att_mfh_2],
																						 feed_dict={test_image: curr_image_feat,
																									test_question: curr_question,
																									test_appendixes:curr_appendixes})
			# pdb.set_trace()
			batch_no += 1
			top_ans = np.argmax(answer_prob, axis=1)
			for i in range(len(answer_prob)):
				# print("...................")
				print(length + 1)
				ans = dataset['ix_to_ans'][str(top_ans[i] + 1)]

				if length <= test_length:
					# print("OE ans ---------", ans)
					open_result.append({u'answer': ans, u'question_id': int(curr_ques_id[i])})
					########## Save Att maps ################
					vis2.append(vis_prob2[i])
					ques2.append(ques_prob2[i])
					model.append(model_att[i])
					vis_mfh_1.append(image_att_mfh_1[i])
					vis_mfh_2.append(image_att_mfh_2[i])
					images_id.append(curr_image_id[i])
					questions_id.append(curr_ques_id[i])
					img_id = curr_image_id[i]
					box = boxes[img_id]
					bbox.append(box)
					
				length += 1
		print("Total OE answers ", len(open_result))
		print("saving results .....")

		with h5py.File("test_dual_cross_guided_att_weights_two_layer.h5", "w") as f:
			f.create_dataset("vis2", data=vis2)
			f.create_dataset("ques2", data=ques2)
			f.create_dataset("model", data=model)
			f.create_dataset("vis_mfh_1", data=vis_mfh_1)
			f.create_dataset("vis_mfh_2", data=vis_mfh_2)
			
			f.create_dataset("image_id",data=images_id)
			f.create_dataset("ques_id", data=questions_id)
			f.create_dataset("boxes", data=bbox)

		with open("vqa_OpenEnded_mscoco_test-dev2015_FMCAN"+ str(curr_epoch)+"_results.json", "w") as f:
			json.dump(open_result, f)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=600,
						help='Batch Size')
	parser.add_argument('--rnn_size', type=int, default=512,
						help='rnn size')
	parser.add_argument('--learning_rate', type=float, default=0.00005,
						help='Batch Size')
	parser.add_argument('--epochs', type=int, default=1000,
						help='Expochs')
	parser.add_argument('--input_embedding_size', type=int, default=300,
						help='word embedding size')
	parser.add_argument('--version', type=int, default=2,
						help='VQA data version')
	parser.add_argument('--num_answers', type=int, default=3000,
						help='output answers nums')
	parser.add_argument('--questoin_max_length', type=int, default=20,
						help='output answers nums')
	parser.add_argument('--data_dir', type=str, default="/home/lmf/mcan_for_vqa2/data",
						help='output answers nums')
	parser.add_argument('--save_path', type=str, default="/disk5/limf/save/dual_cross_guided_att_vqamodel",
						help='save path')
	args = parser.parse_args()
	print(args)
	test(args=args)

if __name__ == '__main__':
	main()