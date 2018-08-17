import tensorflow as tf, numpy as np
import os
import time

import numpy as np
import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import argparse
import data_loader
import dual_cross_guided_att_vqamodel_2
import json
import h5py
import pdb
import sys
sys.path.insert(0, "../")

def train(args):
	loader = data_loader.Data_loader()
	word_embed = loader.get_pre_embedding("data")
	word_embed = np.float32(word_embed)
	model = dual_cross_guided_att_vqamodel_2.Answer_Generator(
		batch_size=args.batch_size,
		dim_image=[6,6,2048],
		dim_hidden=512,
		max_words_q=args.questoin_max_length,
		drop_out_rate_0=0.95,
		drop_out_rate_1=0.9,# constant
		drop_out_rate=0.5, # constant
		num_output=args.num_answers,
		pre_word_embedding=word_embed)
	tf_image, tf_question, tf_label, tf_loss_0,tf_loss_1,tf_loss,tf_predictions,tf_appendixes,tf_scores_emb_0,tf_scores_emb_1,tf_answer,tf_drop_out_rate,tf_drop_out_rate_0,tf_drop_out_rate_1,tf_drop_out_rate_mfh = model.trainer()
	tf_loss_all = tf_loss_0 + tf_loss_1
	####################
	
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.95
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep = 1000)
	global_step = tf.Variable(0)
	sample_size = 600000
	# learning rate decay
	lr = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=sample_size/args.batch_size*2, decay_rate=0.95,
											   staircase=True)

	optimizer = tf.train.AdamOptimizer(lr)
	tvars = tf.trainable_variables()
	model_1_var = tf.get_collection('model_1_var')
	model_2_var = tf.get_collection('model_2_var')
	model_model_var = tf.get_collection('model_model_var')
	# gvs = optimizer.compute_gradients(tf_loss_all, tvars)
	gvs = optimizer.compute_gradients(tf_loss_all, [model_1_var,model_2_var])
	clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs if not grad is None]
	train_op = optimizer.apply_gradients(clipped_gvs, global_step=global_step)
	
	sess.run(tf.global_variables_initializer())
	curr_epoch = 0
	
	"""
	model_file = "/disk5/limf/save/dual_cross_guided_att_vqamodel/model-" + str(curr_epoch)
	print(model_file)
	saver.restore(sess, model_file)
	"""
	"""
	print("Iteration  is done. Saving the model ...")
	saver.save(sess, os.path.join(args.save_path, 'model'), global_step=curr_epoch + 1)
	"""
	# pdb.set_trace()
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
	
	drop_out_rate = 0.6
	drop_out_rate_0=1
	drop_out_rate_1=0.7
	
	test_when_train = True
	# pdb.set_trace()
	for itr in range(curr_epoch + 1, 1000):
		
		batch_no = 0
		# batch_no = 6834
		train_length = len(train_data["questions"])
		all_batch_num = train_length / args.batch_size
		tot_loss = 0.0
		count = 0
		avg_accuracy_0 = 0
		avg_accuracy_1 = 0
		avg_accuracy = 0
		tStart = time.time()
		while (batch_no * args.batch_size) < train_length:
			curr_image_feat, curr_question, curr_answer,appendixes,images_id, questions_id = loader.get_next_batch(batch_no=batch_no,
																		batch_size=args.batch_size,
																		max_question_length=args.questoin_max_length,
																		qa_data=train_data,
																		image_features=image_features_train)
			# if train_length - batch_no * batch_size < batch_size:
			#	 break
			
			_, loss_0,loss_1,loss,pred,scores_emb_0,scores_emb_1,answer = sess.run([train_op, tf_loss_0,tf_loss_1,tf_loss,tf_predictions,tf_scores_emb_0,tf_scores_emb_1,tf_answer], feed_dict={tf_image: curr_image_feat,
															   tf_question: curr_question,
															   # tf_label: np.concatenate([curr_answer,curr_answer],axis=0),
															   tf_label: curr_answer,
															   tf_appendixes:appendixes,
															   tf_drop_out_rate:drop_out_rate,
															   tf_drop_out_rate_0:drop_out_rate_0,
															   tf_drop_out_rate_1:drop_out_rate_1,
																	   tf_drop_out_rate_mfh:0.9})
			
			tot_loss += loss
			lrate = sess.run(lr)
			curr_answer = np.concatenate([curr_answer,curr_answer,curr_answer],axis=0)
			correct_predictions = np.equal(curr_answer,pred)
			correct_predictions_0 = correct_predictions[0:args.batch_size]
			correct_predictions_1 = correct_predictions[args.batch_size:2 * args.batch_size]
			correct_predictions = correct_predictions[2 * args.batch_size:3 * args.batch_size]
			correct_predictions_0 = correct_predictions_0.astype('float32')
			correct_predictions_1 = correct_predictions_1.astype('float32')
			correct_predictions = correct_predictions.astype('float32')
			accuracy_0 = correct_predictions_0.mean()
			accuracy_1 = correct_predictions_1.mean()
			accuracy = correct_predictions.mean()
			if count % 200 == 0:
				print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
				print("Loss:%f batch:%d/%d epoch:%d/%d lrate:%s accuracy_0:%s"%(loss_0,batch_no,all_batch_num,itr,1000,str(lrate),str(accuracy_0)))
				print("Loss:%f accuracy_1:%s"%(loss_1,str(accuracy_1)))
				print("loss:%f accuracy:%s"%(loss,str(accuracy)))

			avg_accuracy_0 += accuracy_0
			avg_accuracy_1 += accuracy_1
			avg_accuracy += accuracy
			count += 1
			
			if count == 300:
				avg_accuracy_0/=count
				avg_accuracy_1/=count
				avg_accuracy/=count
				print ("Acc_all_0", avg_accuracy_0)
				print ("Acc_all_1", avg_accuracy_1)
				print ("Acc_all", avg_accuracy)
				avg_accuracy_0*=count
				avg_accuracy_1*=count
				avg_accuracy*=count
			
			batch_no += 1
			if test_when_train:
				if batch_no % 8 == 0:
					batch_no += 1
			
		avg_accuracy_0/=count
		avg_accuracy_1/=count
		avg_accuracy/=count
		
		sub_0 = avg_accuracy_0 - 0.63
		sub_1 = avg_accuracy_1 - 0.63
		"""
		if sub_0 > 0.02:
			drop_out_rate_0 -=0.05
		else:
			drop_out_rate_0 +=0.05
		
		
		if sub_1 > 0.02:
			drop_out_rate_1 -=0.01
		else:
			drop_out_rate_1 +=0.01
		"""
		if drop_out_rate_0 > 1:
			drop_out_rate_0 = 1
		if drop_out_rate_1 > 1:
			drop_out_rate_1 = 1
		print("drop_out_rate_0:%s"%(str(drop_out_rate_0)))
		print("drop_out_rate_1:%s"%(str(drop_out_rate_1)))
		
		print ("Acc_all_0", avg_accuracy_0)
		print ("Acc_all_1", avg_accuracy_1)
		print ("Acc_all", avg_accuracy)
		
		
		if np.mod(itr, 1) == 0:
			print("Iteration ", itr, " is done. Saving the model ...")
			saver.save(sess, os.path.join(args.save_path, 'model'), global_step=itr)
		tStop = time.time()
		if np.mod(itr, 1) == 0:
			print("Iteration: ", itr, " Loss: ", tot_loss, " Learning Rate: ", lr.eval(session=sess))
			print ("Time Cost:", round(tStop - tStart,2), "s")

		if test_when_train:
			batch_no = 0
			# batch_no = 6834
			train_length = len(train_data["questions"])
			all_batch_num = train_length / args.batch_size
			count = 0
			avg_accuracy_0 = 0
			avg_accuracy_1 = 0
			avg_accuracy = 0
			tStart = time.time()
			while (batch_no * args.batch_size) < train_length:
				if batch_no % 8 == 0:
					curr_image_feat, curr_question, curr_answer,appendixes,images_id, questions_id = loader.get_next_batch(batch_no=batch_no,
																				batch_size=args.batch_size,
																				max_question_length=args.questoin_max_length,
																				qa_data=train_data,
																				image_features=image_features_train)
					
					pred = sess.run([tf_predictions], feed_dict={tf_image: curr_image_feat,
																	   tf_question: curr_question,
																	   tf_label: curr_answer,
																	   tf_appendixes:appendixes,
																	   tf_drop_out_rate:1,
																	   tf_drop_out_rate_0:1,
																	   tf_drop_out_rate_1:1,
																	   tf_drop_out_rate_mfh:1})
					
					curr_answer = np.concatenate([curr_answer,curr_answer,curr_answer],axis=0)
					pred = pred[0]
					correct_predictions = np.equal(curr_answer,pred)
					correct_predictions_0 = correct_predictions[0:args.batch_size]
					correct_predictions_1 = correct_predictions[args.batch_size:2 * args.batch_size]
					correct_predictions = correct_predictions[2 * args.batch_size:3 * args.batch_size]
					correct_predictions_0 = correct_predictions_0.astype('float32')
					correct_predictions_1 = correct_predictions_1.astype('float32')
					correct_predictions = correct_predictions.astype('float32')
					accuracy_0 = correct_predictions_0.mean()
					accuracy_1 = correct_predictions_1.mean()
					accuracy = correct_predictions.mean()
					
					avg_accuracy_0 += accuracy_0
					avg_accuracy_1 += accuracy_1
					avg_accuracy += accuracy
					count += 1
				batch_no += 1
					
			avg_accuracy_0/=count
			avg_accuracy_1/=count
			avg_accuracy/=count
			
			print ("Acc_all_0_test", avg_accuracy_0)
			print ("Acc_all_1_test", avg_accuracy_1)
			print ("Acc_all_test", avg_accuracy)
				
		

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=100,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.00005,
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