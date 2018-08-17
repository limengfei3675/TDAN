import tensorflow as tf, numpy as np
import os
import time

import numpy as np
import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import argparse
import data_loader
import sump_vqamodel_split_ques
import json
import h5py
import pdb
import sys
from random import shuffle
sys.path.insert(0, "../")


def get_next_batch(batch_no, batch_size, max_question_length, qa_data, image_features,shuffle_array):
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
	header = np.zeros((n,max_question_length,1), dtype = 'int32')
	footer = np.zeros((n,max_question_length,1), dtype = 'int32')
	# ---------------
	question = np.ndarray((n, max_question_length), dtype=np.int32)
	count = 0
	# pdb.set_trace()
	questions_id = []
	images_id = []
	
	for i in range(si, ei):
		answer[count] = int(qa_data["answers"][i])
		img_id = int(qa_data["images_id"][i])
		image_feature[count, :] = image_features[img_id]
		question[count, :] = qa_data["questions"][i]
		questions_id.append(int(qa_data["questions_id"][i]))
		images_id.append(img_id)
		
		zero_length = 0
		for step in range(max_question_length):
			if question[count][step] != 0:
				zero_length = step
				break
		for j in range(zero_length,max_question_length):
			appendixes[count][j][j-zero_length] = 1
		header[count][zero_length][0] = 1
		header[count][zero_length + 1][0] = 1
		for j in range(zero_length+2,max_question_length):
			footer[count][j][0] = 1
		count += 1
	
	"""
	for i in range(n):
		r_num = np.random.randint(0, train_length - 1)
		answer[count] = int(qa_data["answers"][r_num])
		img_id = int(qa_data["images_id"][r_num])
		image_feature[count, :] = image_features[img_id]
		question[count, :] = qa_data["questions"][r_num]
		questions_id.append(int(qa_data["questions_id"][r_num]))
		images_id.append(img_id)
		
		zero_length = 0
		for step in range(max_question_length):
			if question[count][step] != 0:
				zero_length = step
				break
		for j in range(zero_length,max_question_length):
			appendixes[count][j][j-zero_length] = 1
		header[count][zero_length][0] = 1
		header[count][zero_length + 1][0] = 1
		for j in range(zero_length+2,max_question_length):
			footer[count][j][0] = 1
		count += 1
	"""
	"""
	while count < n:
		r_num = np.random.randint(0, train_length - 1)
		zero_length = 0
		for step in range(max_question_length):
			if qa_data["questions"][r_num][step] != 0:
				zero_length = step
				break
		if zero_length  > 0:
			answer[count] = int(qa_data["answers"][r_num])
			img_id = int(qa_data["images_id"][r_num])
			image_feature[count, :] = image_features[img_id]
			question[count, :] = qa_data["questions"][r_num]
			questions_id.append(int(qa_data["questions_id"][r_num]))
			images_id.append(img_id)
			
			for j in range(zero_length,max_question_length):
				appendixes[count][j][j-zero_length] = 1
			header[count][zero_length][0] = 1
			header[count][zero_length + 1][0] = 1
			for j in range(zero_length+2,max_question_length):
				footer[count][j][0] = 1
			count += 1
		"""
	return image_feature, question, answer,appendixes,header,footer, images_id, questions_id

def train(args):
	loader = data_loader.Data_loader()
	word_embed = loader.get_pre_embedding("data")
	word_embed = np.float32(word_embed)
	model = sump_vqamodel_split_ques.Answer_Generator(
		batch_size=args.batch_size,
		dim_image=[6,6,2048],
		dim_hidden=1024,
		max_words_q=args.questoin_max_length,
		drop_out_rate_0=0.95,
		drop_out_rate_1=0.9,# constant
		drop_out_rate=0.5, # constant
		num_output=args.num_answers,
		pre_word_embedding=word_embed)
	tf_image, tf_question, tf_label,tf_loss,tf_predictions,tf_appendixes,tf_answer,tf_drop_out_rate,tf_drop_out_rate_mfh,tf_drop_out_rate_1,tf_header,tf_footer = model.trainer()
	# tf_loss_all = tf_loss_0 + tf_loss_1
	####################
	
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep = 1000)
	global_step = tf.Variable(0)
	sample_size = 600000
	# learning rate decay
	lr = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=sample_size/args.batch_size*2, decay_rate=0.95,
											   staircase=True)

	optimizer = tf.train.AdamOptimizer(lr)
	# optimizer = tf.train.AdagradOptimizer(lr)
	tvars = tf.trainable_variables()

	gvs = optimizer.compute_gradients(tf_loss, tvars)
	clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs if not grad is None]
	train_op = optimizer.apply_gradients(clipped_gvs, global_step=global_step)
	
	sess.run(tf.global_variables_initializer())
	curr_epoch = 0
	"""
	model_file = "/disk5/limf/save/dual_cross_guided_att_vqamodel/model-" + str(curr_epoch)
	print(model_file)
	saver.restore(sess, model_file)
	"""
	
	print("Loading data...............")
	dataset, train_data = loader.get_qa_data(args.data_dir, data_set="train")
	
	print("Loading image features ...")
	image_features_train = loader.get_image_feature(train=True)
	# image_features_test = loader.get_image_feature(train=False)
	print("Done !")
	# print("Image data length is %d " %len(image_features))
	word_num = len(dataset["ix_to_word"])
	print("Vocab size is %d "%word_num)
	print("Answer num is %d "%len(dataset["ix_to_ans"]))
	print("Loading pre-word-embedding ......")
	
	drop_out_rate = 0.5
	drop_out_rate_mfh=0.9
	drop_out_rate_1=1
	top_accu = 0
	test_when_train = False
	# pdb.set_trace()
	for itr in range(curr_epoch + 1, 1000):
		
		batch_no = 0
		# batch_no = 6834
		train_length = len(train_data["questions"])
		all_batch_num = train_length / args.batch_size
		tot_loss = 0.0
		count = 0
		avg_accuracy = 0
		train_batch_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,15]
		count_test = 0
		avg_accuracy_test = 0
		test_batch_list = [0,15]
		shuffle_array = range(train_length)
		shuffle_array = shuffle(shuffle_array)
		tStart = time.time()
		while (batch_no * args.batch_size) < train_length:
			if(batch_no % 16 in train_batch_list):
				curr_image_feat, curr_question, curr_answer,appendixes,header,footer,images_id, questions_id = get_next_batch(batch_no=batch_no,
																			batch_size=args.batch_size,
																			max_question_length=args.questoin_max_length,
																			qa_data=train_data,
																			image_features=image_features_train,
																			shuffle_array = shuffle_array)
				# if train_length - batch_no * batch_size < batch_size:
				#	 break
				
				_,loss,pred,answer = sess.run([train_op,tf_loss,tf_predictions,tf_answer], feed_dict={tf_image: curr_image_feat,
																   tf_question: curr_question,
																   tf_label: curr_answer,
																   tf_appendixes:appendixes,
																   tf_drop_out_rate:drop_out_rate,
																   tf_drop_out_rate_mfh:drop_out_rate_mfh,
																   tf_drop_out_rate_1:drop_out_rate_1,
																   tf_header:header,
																   tf_footer:footer})
				
				tot_loss += loss
				lrate = sess.run(lr)
				correct_predictions = np.equal(curr_answer,pred)
				correct_predictions = correct_predictions.astype('float32')
				accuracy = correct_predictions.mean()
				if count % 400 == 0:
					print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
					print("Loss:%f batch:%d/%d epoch:%d/%d lrate:%s accuracy:%s"%(loss,batch_no,all_batch_num,itr,1000,str(lrate),str(accuracy)))
				avg_accuracy += accuracy
				count += 1
				
				if count == 300:
					avg_accuracy/=count
					print ("Acc_all", avg_accuracy)
					avg_accuracy*=count
			else:
				if test_when_train:
					if(batch_no % 16 in test_batch_list):
						curr_image_feat, curr_question, curr_answer,appendixes,header,footer,images_id, questions_id = get_next_batch(batch_no=batch_no,
																					batch_size=args.batch_size,
																					max_question_length=args.questoin_max_length,
																					qa_data=train_data,
																					image_features=image_features_train,
																					shuffle_array = shuffle_array)
						
						pred = sess.run([tf_predictions], feed_dict={tf_image: curr_image_feat,
																		   tf_question: curr_question,
																		   tf_label: curr_answer,
																		   tf_appendixes:appendixes,
																			tf_drop_out_rate:1,
																			tf_drop_out_rate_mfh:1,
																			tf_drop_out_rate_1:1,
																			tf_header:header,
																			tf_footer:footer})
						
						pred = pred[0]
						correct_predictions = np.equal(curr_answer,pred)
						correct_predictions = correct_predictions.astype('float32')
						accuracy = correct_predictions.mean()
						
						avg_accuracy_test += accuracy
						count_test += 1
			batch_no += 1
				
		avg_accuracy/=count
		
		sub = avg_accuracy - 0.7
		if sub > 0:
			drop_out_rate_1 -= 0.1
		else:
			drop_out_rate_1 += 0.1
		
		if drop_out_rate_1 > 1:
			drop_out_rate_1 = 1
		print("drop_out_rate_1:%s"%(str(drop_out_rate_1)))
		
		print ("Acc_all", avg_accuracy)
		
		if not test_when_train:
			print("Iteration ", itr, " is done. Saving the max model ...")
			saver.save(sess, os.path.join(args.save_path, 'model'), global_step=itr)
		else:
			avg_accuracy_test/=count_test
			print ("Acc_all_test", avg_accuracy_test)
			if avg_accuracy_test > top_accu:
				top_accu = avg_accuracy_test
				print("Iteration ", itr, " is done. Saving the max model ...")
				saver.save(sess, os.path.join(args.save_path, 'model'), global_step=itr)
		
				
def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=100,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.0005,
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
	parser.add_argument('--save_path', type=str, default="/disk5/limf/save/dual_cross_guided_att_vqamodel",
						help='save path')
	args = parser.parse_args()
	print(args)
	train(args=args)

if __name__ == '__main__':
	main()