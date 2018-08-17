import tensorflow as tf, numpy as np
import os
import time

import numpy as np
import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import argparse
import data_loader
import dual_cross_guided_att_vqamodel
import json
import h5py
import pdb
import sys
sys.path.insert(0, "../")

def train(args):
	loader = data_loader.Data_loader()
	word_embed = loader.get_pre_embedding("data")
	word_embed = np.float32(word_embed)
	model = dual_cross_guided_att_vqamodel.Answer_Generator(
		batch_size=args.batch_size,
		dim_image=[6,6,2048],
		dim_hidden=512,
		dim_attention=512,
		max_words_q=args.questoin_max_length,
		drop_out_rate=0.61,
		num_output=args.num_answers,
		pre_word_embedding=word_embed)
	tf_image, tf_question, tf_label, tf_loss,tf_predictions,tf_appendixes = model.trainer()
	# pdb.set_trace()
	tf_question_r,tf_label_r,tf_appendixes_r,tf_loss_r,tf_predictions_r,model_att_out,cross_entropy = model.trainer_r()
	####################
	
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep = 1000)
	global_step = tf.Variable(0)
	sample_size = 600000
	# learning rate decay
	lr = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=sample_size/args.batch_size*2, decay_rate=0.95,
											   staircase=True)

	optimizer = tf.train.AdamOptimizer(lr)
	tvars = tf.trainable_variables()
	gvs = optimizer.compute_gradients(tf_loss, tvars)
	clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs if not grad is None]
	train_op = optimizer.apply_gradients(clipped_gvs, global_step=global_step)
	
	gvs_r = optimizer.compute_gradients(tf_loss_r, tvars)
	clipped_gvs_r = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs_r if not grad is None]
	train_op_r = optimizer.apply_gradients(clipped_gvs_r, global_step=global_step)
	
	sess.run(tf.global_variables_initializer())
	curr_epoch = 193
	
	model_file = "save/dual_cross_guided_att_vqamodel/model-" + str(curr_epoch)
	print(model_file)
	saver.restore(sess, model_file)
	
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
	
	# pdb.set_trace()
	for itr in range(curr_epoch + 1, 1000):
		
		
		batch_no = 0
		# batch_no = 6834
		train_length = len(train_data["questions"])
		all_batch_num = train_length / args.batch_size
		sum_step = 0
		tot_loss = 0.0
		avg_accuracy_0 = 0
		avg_accuracy_1 = 0
		avg_accuracy_r = 0
		tStart = time.time()
		while (batch_no * args.batch_size) < train_length:
			curr_image_feat, curr_question, curr_answer,appendixes = loader.get_next_batch(batch_no=batch_no,
																		batch_size=args.batch_size,
																		max_question_length=args.questoin_max_length,
																		qa_data=train_data,
																		image_features=image_features_train)
			# if train_length - batch_no * batch_size < batch_size:
			#	 break
			
			_, loss,pred = sess.run([train_op, tf_loss,tf_predictions], feed_dict={tf_image: curr_image_feat,
															   tf_question: curr_question,
															   tf_label: np.concatenate([curr_answer,curr_answer],axis=0),
															   tf_appendixes:appendixes})
			
			# sum_writer.add_summary(summary, sum_step)
			sum_step += 1
			
			tot_loss += loss
			lrate = sess.run(lr)
			curr_answer = np.concatenate([curr_answer,curr_answer],axis=0)
			correct_predictions = np.equal(curr_answer,pred)
			correct_predictions_0 = correct_predictions[0:args.batch_size]
			correct_predictions_1 = correct_predictions[args.batch_size:2 * args.batch_size]
			correct_predictions_0 = correct_predictions_0.astype('float32')
			correct_predictions_1 = correct_predictions_1.astype('float32')
			accuracy_0 = correct_predictions_0.mean()
			accuracy_1 = correct_predictions_1.mean()
			sum_0 = accuracy_0 * args.batch_size
			sum_1 = accuracy_1 * args.batch_size
			sub = sum_0 - sum_1
			correct_predictions_reshape = np.concatenate([np.expand_dims(correct_predictions_0, axis=1),np.expand_dims(correct_predictions_1, axis=1)],1)
			
			"""
			aaaaa = np.ndarray((args.batch_size, 2), dtype=np.float32)
			for i in range(args.batch_size):
				for j in range(2):
					aaaaa[i][j] = correct_predictions_reshape[i][j]
			sub_aaa = sub
			"""
			
			"""
			if sub > 0:
				for i in range(args.batch_size):
					if correct_predictions_reshape[i][0] == correct_predictions_reshape[i][1]:
						if sub > 0:
							correct_predictions_reshape[i][0] = 0
							correct_predictions_reshape[i][1] = 1
							sub -= 1
						else:
							break
			
							if (i+batch_no) % 2 == 0:
								correct_predictions_reshape[i][0] = 0
								correct_predictions_reshape[i][1] = 1
							else:
								correct_predictions_reshape[i][0] = 1
								correct_predictions_reshape[i][1] = 0
							
			elif sub < 0:
				for i in range(args.batch_size):
					if correct_predictions_reshape[i][0] == correct_predictions_reshape[i][1]:
						if sub < 0:
							correct_predictions_reshape[i][0] = 1
							correct_predictions_reshape[i][1] = 0
							sub += 1
						else:
							break
							
							if (i+batch_no) % 2 == 0:
								correct_predictions_reshape[i][0] = 0
								correct_predictions_reshape[i][1] = 1
							else:
								correct_predictions_reshape[i][0] = 1
								correct_predictions_reshape[i][1] = 0
							
			
			else:
				for i in range(args.batch_size):
					if (i+batch_no) % 2 == 0:
						correct_predictions_reshape[i][0] = 0
						correct_predictions_reshape[i][1] = 1
					else:
						correct_predictions_reshape[i][0] = 1
						correct_predictions_reshape[i][1] = 0

			for i in range(args.batch_size):
				if correct_predictions_reshape[i][0] == correct_predictions_reshape[i][1]:
					print(".....................")
					print(correct_predictions_reshape[i][0],correct_predictions_reshape[i][1])
					pdb.set_trace()
			"""	
			_, loss_r,pred_r = sess.run([train_op_r, tf_loss_r,tf_predictions_r], feed_dict={tf_question_r: curr_question,
															   tf_label_r:correct_predictions_reshape,
															   tf_appendixes_r:appendixes})
			# pdb.set_trace()
			labal_r = np.argmax(correct_predictions_reshape,1)
			correct_predictions_r = np.equal(labal_r,pred_r)
			correct_predictions_r = correct_predictions_r.astype('float32')
			accuracy_r = correct_predictions_r.mean()
			
			if batch_no % 30 == 0:
				print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
				print("Loss:%f batch:%d/%d epoch:%d/%d learning rate:%s accuracy_0:%s"%(loss,batch_no,all_batch_num,itr,1000,str(lrate),str(accuracy_0)))
				print("Loss:%f batch:%d/%d epoch:%d/%d learning rate:%s accuracy_1:%s"%(loss,batch_no,all_batch_num,itr,1000,str(lrate),str(accuracy_1)))
				print("loss_r:%f batch:%d/%d epoch:%d/%d learning rate:%s accuracy_r:%s"%(loss_r,batch_no,all_batch_num,itr,1000,str(lrate),str(accuracy_r)))

		
			avg_accuracy_0 += accuracy_0
			avg_accuracy_1 += accuracy_1
			avg_accuracy_r += accuracy_r
			batch_no += 1
		avg_accuracy_0/=batch_no
		avg_accuracy_1/=batch_no
		avg_accuracy_r/=batch_no
		print ("Acc_all_0", avg_accuracy_0)
		print ("Acc_all_1", avg_accuracy_1)
		print ("Acc_all_r", avg_accuracy_r)
		tStop = time.time()
		if np.mod(itr, 1) == 0:
			print("Iteration: ", itr, " Loss: ", tot_loss, " Learning Rate: ", lr.eval(session=sess))
			print ("Time Cost:", round(tStop - tStart,2), "s")
		if np.mod(itr, 1) == 0:
			print("Iteration ", itr, " is done. Saving the model ...")
			saver.save(sess, os.path.join(args.save_path, 'model'), global_step=itr)

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
	train(args=args)

if __name__ == '__main__':
	main()