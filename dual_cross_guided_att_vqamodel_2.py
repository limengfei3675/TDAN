import csv
import pdb

import tensorflow as tf
import numpy as np
rnn_cell = tf.nn.rnn_cell
import sys

csv.field_size_limit(sys.maxsize)

class Answer_Generator():

	

	def __init__(self,batch_size,dim_image,dim_hidden
					   ,max_words_q, drop_out_rate_0,drop_out_rate_1,drop_out_rate, num_output, pre_word_embedding):
		print("Initializing dual cross-guided two-layer vqa model.........")
		self.batch_size = batch_size
		self.dim_image = dim_image
		self.dim_hidden = dim_hidden
		self.dim_q = dim_hidden
		self.max_words_q = max_words_q
		"""
		drop_out_rate_0 = drop_out_rate_0
		drop_out_rate_1 = drop_out_rate_1
		drop_out_rate = drop_out_rate
		"""
		self.num_output= num_output
		self.K = 36
		self.hid = dim_hidden
		self.lamb = 10e-8
		self.model_num = 2
		# question-embedding
		# self.embed_question = tf.Variable(
		#	 tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_question')
		self.pre_word_embedding = pre_word_embedding
		self.num_img_glimpse_coatt = 2
		self.num_ques_glimpse_coatt = 2
		# image-embedding

		self.embed_image_W = self.get_weights(w_shape=[self.dim_image[2], self.dim_hidden], name="embed_image_W", lamb= self.lamb,train=True)
		self.embed_image_b =self.get_bias(b_shape=[self.dim_hidden], name="embed_image_b",train=True)
		tf.add_to_collection("model_1_var", self.embed_image_W)
		tf.add_to_collection("model_1_var", self.embed_image_b)
		
		self.embed_ques_W = self.get_weights(w_shape=[self.dim_q, self.dim_hidden], name="embed_ques_W", lamb=self.lamb,train=True)
		self.embed_ques_b = self.get_bias(b_shape=[self.dim_hidden], name="embed_ques_b",train=True)
		tf.add_to_collection("model_1_var", self.embed_ques_W)
		tf.add_to_collection("model_1_var", self.embed_ques_b)
		
		self.img_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="img_att_W", lamb=self.lamb,train=True)
		self.img_att_b = self.get_bias(b_shape=[1], name="img_att_b",train=True)
		tf.add_to_collection("model_1_var", self.img_att_W)
		tf.add_to_collection("model_1_var", self.img_att_b)

		self.ques_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="ques_att_W", lamb=self.lamb,train=True)
		self.ques_att_b = self.get_bias(b_shape=[1], name="ques_att_b",train=True)
		tf.add_to_collection("model_1_var", self.ques_att_W)
		tf.add_to_collection("model_1_var", self.ques_att_b)
		
		self.qa_W_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_clf", lamb=self.lamb,train=True)
		self.qa_b_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_img",train=True)
		tf.add_to_collection("model_1_var", self.qa_W_clf)
		tf.add_to_collection("model_1_var", self.qa_b_clf)
		
		self.qa_W_prime_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_prime_clf", lamb=self.lamb,train=True)
		self.qa_b_prime_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_clf",train=True)
		tf.add_to_collection("model_1_var", self.qa_W_prime_clf)
		tf.add_to_collection("model_1_var", self.qa_b_prime_clf)
		# score-embedding
		self.embed_scor_W = self.get_weights(w_shape=[self.dim_hidden, self.num_output], name="embed_scor_W", lamb=self.lamb,train=True)
		self.embed_scor_b = self.get_bias(b_shape=[self.num_output], name="embed_scor_b",train=True)
		tf.add_to_collection("model_1_var", self.embed_scor_W)
		tf.add_to_collection("model_1_var", self.embed_scor_b)
		
		# score-embedding
		self.embed_scor_W_mfh = self.get_weights(w_shape=[self.dim_hidden * 2, self.num_output], name="embed_scor_W_mfh", lamb=self.lamb)
		self.embed_scor_b_mfh = self.get_bias(b_shape=[self.num_output], name="embed_scor_b_mfh")
		tf.add_to_collection("model_2_var", self.embed_scor_W_mfh)
		tf.add_to_collection("model_2_var", self.embed_scor_b_mfh)
		
		self.feature_dim = self.dim_hidden/2
		self.conv1d_kernel_11,self.conv1d_b_11 = self.conv1d_param(300 + self.max_words_q,1,self.feature_dim)
		self.conv1d_kernel_12,self.conv1d_b_12 = self.conv1d_param(300 + self.max_words_q,3,self.feature_dim/2)
		self.conv1d_kernel_13,self.conv1d_b_13 = self.conv1d_param(300 + self.max_words_q,5,self.feature_dim/2)
		tf.add_to_collection("model_model_var", self.conv1d_kernel_11)
		tf.add_to_collection("model_model_var", self.conv1d_b_11)
		tf.add_to_collection("model_model_var", self.conv1d_kernel_12)
		tf.add_to_collection("model_model_var", self.conv1d_b_12)
		tf.add_to_collection("model_model_var", self.conv1d_kernel_13)
		tf.add_to_collection("model_model_var", self.conv1d_b_13)
		
		
		self.model_att_hidden_w = self.get_weights(w_shape=[self.dim_q, self.dim_hidden],name="model_att_hidden_w", lamb=self.lamb)
		self.model_att_hidden_b = self.get_bias(b_shape=[self.dim_hidden], name="model_att_hidden_b")
		tf.add_to_collection("model_model_var", self.model_att_hidden_w)
		tf.add_to_collection("model_model_var", self.model_att_hidden_b)
		
		self.model_att_w = self.get_weights(w_shape=[self.dim_hidden, self.model_num],name="model_att_w", lamb=self.lamb)
		self.model_att_b = self.get_bias(b_shape=[self.model_num], name="model_att_b")
		tf.add_to_collection("model_model_var", self.model_att_w)
		tf.add_to_collection("model_model_var", self.model_att_b)
		
	# cnn
	def conv1d_param(self,input_tensor_length, kernel_size, feature):
		
		kernel = tf.Variable(tf.truncated_normal(shape=[kernel_size, input_tensor_length, feature*2], stddev=0.1), name="W")
		weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.lamb)
		tf.add_to_collection("losses", weight_decay)
		tf.add_to_collection("losses_1", weight_decay)
		tf.add_to_collection("losses_2", weight_decay)
		b = tf.Variable([feature*2], 
						dtype=tf.float32)
		return kernel,b
	
	def get_weights(self, name, w_shape, lamb,train = True):
		if train:
			weight = tf.Variable(tf.random_uniform(w_shape, -0.01, 0.01),name=name)
		else:
			weight = tf.random_uniform(w_shape, -0.01, 0.01)
		weight_decay = tf.multiply(tf.nn.l2_loss(weight), lamb)
		tf.add_to_collection("losses", weight_decay)
		tf.add_to_collection("losses_1", weight_decay)
		tf.add_to_collection("losses_2", weight_decay)
		return weight

	def get_bias(self, name, b_shape,train = True):
		if train:
			bias = tf.Variable(tf.random_uniform(b_shape, -0.08, 0.08),name=name)
		else:
			bias = tf.random_uniform(b_shape, -0.08, 0.08)
		return bias

	#conv1d
	def conv1d_layer(self,input_tensor,conv1d_kernel, conv1d_b,feature):
		
		output_conv1d = tf.nn.conv1d(input_tensor, conv1d_kernel, 1, 'SAME') + conv1d_b
		
		conv1_half_1 = output_conv1d[:,:,0:feature]
		conv1_half_2 = output_conv1d[:,:,feature:feature*2]
		# pdb.set_trace()
		conv1_half_2 = tf.nn.softmax(conv1_half_2)
		conv1_half_2 = conv1_half_1 * conv1_half_2
		
		output = conv1_half_2
		
		return output
		
	def forward_pass_cnn(self, tensor_x,train_val):
		output_0 = self.conv1d_layer(tensor_x,self.conv1d_kernel_11,self.conv1d_b_11,self.feature_dim)
		output_1 = self.conv1d_layer(tensor_x,self.conv1d_kernel_12,self.conv1d_b_12,self.feature_dim/2)
		output_2 = self.conv1d_layer(tensor_x,self.conv1d_kernel_13,self.conv1d_b_13,self.feature_dim/2)
		output = tf.concat([output_0,output_1,output_2],axis=2)
		"""
		res_output = tf.nn.conv1d(tensor_x, self.conv1d_kernel_21, 1, 'SAME') + self.conv1d_b_21
		output = res_output + output
		"""
		# output11_2 = tf.matmul(tensor_x,self.liner_transf_W_64_32) + self.liner_transf_b_64_32
		# output12 = output12 + output11_2
		# pooling_1=tf.nn.max_pool(tf.reshape(output12,shape = [self.batch_size,1,self.max_words_q,self.dim_q]),[1,1,self.max_words_q,1],[1,1,1,1],padding='VALID')
	
		return output
		
	def pre_model(self):
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q]) 
		appendixes = tf.placeholder('float32', [self.batch_size, self.max_words_q,self.max_words_q], name = "appendixes")
		drop_out_rate = tf.placeholder(tf.float32)
		word_embeddings = []
		for i in range(self.max_words_q):
			word_emb = tf.nn.embedding_lookup(self.pre_word_embedding, question[:,i])
			word_emb = tf.nn.dropout(word_emb, drop_out_rate, name = "word_emb" + str(i))
			word_embeddings.append(word_emb)

		word_embeddings = tf.transpose(word_embeddings, perm=[1, 0, 2])
		# pdb.set_trace()
		word_embeddings = tf.concat([word_embeddings,appendixes],axis=2)
		tensor_x = tf.convert_to_tensor(word_embeddings)
		# tensor_x = tf.nn.dropout(tensor_x, keep_prob=drop_out_rate_0)
		output = self.forward_pass_cnn(tensor_x,'train')
		return question,output,appendixes,drop_out_rate

	def model(self):

		image = tf.placeholder(tf.float32, [self.batch_size, self.K, self.dim_image[2]]) # b*36*2048
		label = tf.placeholder(tf.int64, [self.batch_size, ]) # b
		drop_out_rate_0 = tf.placeholder(tf.float32)
		drop_out_rate_1 = tf.placeholder(tf.float32)
		drop_out_rate_mfh = tf.placeholder(tf.float32)
		# drop_out = tf.placeholder(tf.float32)
		"""
		question_feat, _ = self.rnn_model.bi_gru_question(self.batch_size, pre_word_embedding=self.pre_word_embedding,
												  inputs=question, time_step=self.max_words_q,
												  layer_num=1, hidden_size=self.rnn_size)   # b*26*1024
		"""
		question,output,appendixes,drop_out_rate = self.pre_model()
		
		
		image_feat = tf.nn.l2_normalize(image, -1)
		# embedding
		image_emb = tf.reshape(image_feat, [-1, self.dim_image[2]])  # (b x m) x d
		image_emb = tf.nn.dropout(image_emb, keep_prob=drop_out_rate)
		image_emb = tf.nn.xw_plus_b(image_emb,self.embed_image_W, self.embed_image_b)
		image_emb = tf.tanh(tf.reshape(image_emb, shape=[self.batch_size, self.K, self.dim_hidden])) # (b*6*6)*2048

		ques_emb = tf.reshape(output, [-1, self.dim_q])
		ques_emb = tf.nn.dropout(ques_emb, keep_prob=drop_out_rate)
		ques_emb = tf.nn.xw_plus_b(ques_emb, self.embed_ques_W, self.embed_ques_b)
		ques_emb = tf.tanh(tf.reshape(ques_emb, shape=[self.batch_size,-1, self.dim_hidden]))  # (b*26)*1024

		# first layer attention
		# image_emb = tf.nn.dropout(image_emb, keep_prob=drop_out_rate_0)
		image_emb_att = tf.nn.xw_plus_b(tf.reshape(image_emb, shape=[-1,self.dim_hidden]), self.img_att_W, self.img_att_b)
		self.image_emb_prob = tf.nn.softmax(tf.reshape(image_emb_att, shape=[self.batch_size, -1]))
		image_emb_prob = self.image_emb_prob
		img_memory = tf.reduce_sum(tf.expand_dims(image_emb_prob,2)*image_emb, axis=1)
		# reduce
		# ques_emb = tf.nn.dropout(ques_emb, keep_prob=drop_out_rate_0)
		ques_emb_att = tf.nn.xw_plus_b(tf.reshape(ques_emb, shape=[-1,self.dim_hidden]), self.ques_att_W, self.ques_att_b)
		ques_emb_prob = tf.nn.softmax(tf.reshape(ques_emb_att, shape=[self.batch_size, -1]))
		ques_memory = tf.reduce_sum(tf.expand_dims(ques_emb_prob, 2) * ques_emb, axis=1)
		# ques * image
		memory = img_memory*ques_memory  # b*1024
		# # attention models
		with tf.variable_scope("att1"):
			# vis_comb1 512   ques_comb1 1024
			self.vis_att_prob1, vis_comb1 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb,drop_out_rate_0=drop_out_rate_0)
			self.ques_att_prob1, ques_comb1 =self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb,drop_out_rate_0=drop_out_rate_0)
			img_memory = img_memory + vis_comb1
			ques_memory = ques_memory + ques_comb1
			memory = memory + img_memory*ques_memory
		with tf.variable_scope("att2"):
			self.vis_att_prob2, vis_comb2 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb,drop_out_rate_0=drop_out_rate_0)
			self.ques_att_prob2, ques_comb2 = self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb,drop_out_rate_0=drop_out_rate_0)
			img_memory = img_memory + vis_comb2
			ques_memory = ques_memory + ques_comb2
			memory = memory + img_memory * ques_memory

		s_head = self.gated_tanh(memory, self.qa_W_clf, self.qa_b_clf, self.qa_W_prime_clf, self.qa_b_prime_clf,drop_out_rate_0)
		# s_head = memory * memory * memory
		s_head = tf.nn.dropout(s_head, keep_prob=drop_out_rate)
		"""
		begin
		"""
		
		"""
		question_emb = tf.reduce_max(output, 1)
		#attention models
		with tf.variable_scope("san_att1"):
			san_prob_att1, comb_emb = self.attention_san(question_emb, image_emb_2d)
		with tf.variable_scope("san_att2"):
			san_prob_att2, comb_emb = self.attention_san(comb_emb, image_emb_2d)

		# head = model_att_out_0 * s_head + model_att_out_1 * comb_emb
		s_head = tf.nn.dropout(s_head, keep_prob=drop_out_rate)
		comb_emb = tf.nn.dropout(comb_emb, keep_prob=drop_out_rate)
		"""
		comb_emb = self.co_att_MFH(output,image,drop_out_rate_1,drop_out_rate_mfh)
		comb_emb = tf.nn.dropout(comb_emb, keep_prob=drop_out_rate)
		"""
		end
		"""
		
		scores_emb_0 = tf.nn.xw_plus_b(s_head, self.embed_scor_W, self.embed_scor_b)
		scores_emb_1 = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W_mfh, self.embed_scor_b_mfh)
		"""
		begin
		"""
		# scores_emb = tf.tanh(scores_emb)
		"""
		end
		"""
		print("classification nums")
		
		question_emb = tf.reduce_max(output, 1)
		question_emb = tf.nn.dropout(question_emb, keep_prob=drop_out_rate)
		# pdb.set_trace()
		model_att_hidden = tf.nn.xw_plus_b(question_emb, self.model_att_hidden_w, self.model_att_hidden_b)
		
		model_att_hidden =  tf.nn.relu(model_att_hidden)
		model_att_hidden = tf.nn.dropout(model_att_hidden, keep_prob=drop_out_rate)
		model_att_out = tf.nn.xw_plus_b(model_att_hidden, self.model_att_w, self.model_att_b)
		
		model_att_out =  tf.nn.softmax(model_att_out)
		self.model_att = model_att_out
		model_att_out_0 = model_att_out[:,0:1]
		model_att_out_1 = model_att_out[:,1:2]
		# answer = model_att_out_0 * scores_emb_0 + model_att_out_1 * scores_emb_1
		answer =  scores_emb_0 + scores_emb_1
		return image, question, label, scores_emb_0,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh

	def trainer(self):
		image, question, label, scores_emb_0,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh = self.model()
		scores_emb = tf.concat([scores_emb_0,scores_emb_1,answer], axis=0)
		# label = tf.concat([label,label], axis=0)
		# Calculate cross entropy
		cross_entropy_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=scores_emb_0)
		# Calculate loss
		loss_0 = tf.reduce_mean(cross_entropy_0)
		tf.add_to_collection('losses_0', loss_0)
		loss_0 = tf.add_n(tf.get_collection('losses_0'), name='total_loss_0')
		
		# Calculate cross entropy
		cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=scores_emb_1)
		# Calculate loss
		loss_1 = tf.reduce_mean(cross_entropy_1)
		tf.add_to_collection('losses_1', loss_1)
		loss_1 = tf.add_n(tf.get_collection('losses_1'), name='total_loss_1')
		
		# Calculate cross entropy
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=answer)
		# Calculate loss
		loss = tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses', loss)
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		predictions = tf.argmax(scores_emb,1)
		
		# pdb.set_trace()
		return image,question,label,loss_0,loss_1,loss,predictions,appendixes,scores_emb_0,scores_emb_1,answer,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh

	
	def solver(self):
		image, question, label, scores_emb_0,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh = self.model()
		answer = tf.nn.softmax(answer)
		scores_emb_0 = tf.nn.softmax(scores_emb_0)
		scores_emb_1 = tf.nn.softmax(scores_emb_1)
		return image, question,scores_emb_0,scores_emb_1,answer,appendixes,self.vis_att_prob2,self.ques_att_prob2,self.model_att,self.image_att_mfh_1,self.image_att_mfh_2,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh

	def get_three_out(self):
		image, question, label, scores_emb_0,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh = self.model()
		answer = tf.nn.softmax(answer)
		answer_0 = tf.nn.softmax(scores_emb_0)
		answer_1 = tf.nn.softmax(scores_emb_1)
		return image, question, answer,answer_0,answer_1,appendixes,drop_out_rate,drop_out_rate_0,drop_out_rate_1,drop_out_rate_mfh
	def gated_tanh(self, concated, w1, b1, w2, b2,drop_out_rate_0):
		concated = tf.nn.dropout(concated, keep_prob=drop_out_rate_0)
		y_tilde = tf.tanh(tf.nn.xw_plus_b(concated, w1, b1))
		g = tf.sigmoid(tf.nn.xw_plus_b(concated, w2, b2))
		y = tf.multiply(y_tilde, g)
		return y

	def tanh_vis_attention(self, question_emb, image_emb,drop_out_rate_0):
		# Attention weight
		# question_emb b*1024  image_emb b*K*2048
		# question-attention
		# probability-attention
		gt_W_img_att = self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_img_att", lamb=self.lamb)
		gt_b_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_img_att")
		tf.add_to_collection("model_1_var", gt_W_img_att)
		tf.add_to_collection("model_1_var", gt_b_img_att)
		
		gt_W_prime_img_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_img_att", lamb=self.lamb)
		gt_b_prime_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_img_att")
		tf.add_to_collection("model_1_var", gt_W_prime_img_att)
		tf.add_to_collection("model_1_var", gt_b_prime_img_att)
		
		prob_image_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_image_att_W", lamb=self.lamb)
		prob_image_att_b = self.get_bias(b_shape=[1], name="prob_image_att_b")
		tf.add_to_collection("model_1_var", prob_image_att_W)
		tf.add_to_collection("model_1_var", prob_image_att_b)
		# pdb.set_trace()
		qenc_reshape = tf.tile(tf.expand_dims(question_emb, 1), multiples=[1, self.K, 1])  # b * k * 1024
		concated = tf.concat([image_emb, qenc_reshape], axis=2)  # b * m * (image_dim + ques_dim)
		concated = tf.reshape(concated, shape=[self.batch_size * self.K, -1])
		concated = self.gated_tanh(concated, gt_W_img_att, gt_b_img_att, gt_W_prime_img_att, gt_b_prime_img_att,drop_out_rate_0)  # (b * m) * hid
		concated = tf.nn.dropout(concated, keep_prob=drop_out_rate_0)
		att_map = tf.nn.xw_plus_b(concated, prob_image_att_W, prob_image_att_b)  # b*m*1
		att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.K]))
		v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * image_emb, axis=1)
		return att_prob, v_head

	def tanh_ques_attention(self, question_emb, image_emb,drop_out_rate_0):
		# Attention weight
		gt_W_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_ques_att", lamb=self.lamb)
		gt_b_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_ques_att")
		tf.add_to_collection("model_1_var", gt_W_ques_att)
		tf.add_to_collection("model_1_var", gt_b_ques_att)
		
		gt_W_prime_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_ques_att", lamb=self.lamb)
		gt_b_prime_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_ques_att")
		tf.add_to_collection("model_1_var", gt_W_prime_ques_att)
		tf.add_to_collection("model_1_var", gt_b_prime_ques_att)
		
		prob_ques_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_ques_att_W", lamb=self.lamb)
		prob_ques_att_b = self.get_bias(b_shape=[1], name="prob_ques_att_b")
		tf.add_to_collection("model_1_var", prob_ques_att_W)
		tf.add_to_collection("model_1_var", prob_ques_att_b)
		
		img_reshape = tf.tile(tf.expand_dims(image_emb, 1), multiples=[1, self.max_words_q, 1])  # b * 26 * 1024
		concated = tf.concat([question_emb, img_reshape], axis=2)  # b * 26 * (image_dim + ques_dim)
		concated = tf.reshape(concated, shape=[self.batch_size * self.max_words_q, -1])
		concated = self.gated_tanh(concated, gt_W_ques_att, gt_b_ques_att, gt_W_prime_ques_att, gt_b_prime_ques_att,drop_out_rate_0)  # (b * m) * hid
		concated = tf.nn.dropout(concated, keep_prob=drop_out_rate_0)
		att_map = tf.nn.xw_plus_b(concated, prob_ques_att_W, prob_ques_att_b)  # b*m*1
		att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.max_words_q]))
		v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * question_emb, axis=1)
		return att_prob, v_head

	def MFH(self,tensor_1,tensor_2,dim_middle,dim_out):
		
		dim_1 = tensor_1.shape.as_list()[2]
		dim_2 = tensor_2.shape.as_list()[2]
		self.mfb_transf_1,_ = self.conv1d_param(dim_1,1,dim_middle/4)
		self.mfb_transf_2,_ = self.conv1d_param(dim_2,1,dim_middle/4)
		tensor_1_transf = tf.nn.conv1d(tensor_1, self.mfb_transf_1, stride=1,padding='SAME')
		tensor_2_transf = tf.nn.conv1d(tensor_2, self.mfb_transf_2, stride=1,padding='SAME')
		tensor_mix_1 = tensor_1_transf * tensor_2_transf
		tensor_mix_1 = tf.nn.dropout(tensor_mix_1,drop_out_rate_1)
		pooling_rate = dim_middle / dim_out
		pooling_1=tf.nn.max_pool(tf.reshape(tensor_mix_1,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		# pdb.set_trace()
		normal_out_1 = tf.reshape(pooling_1,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_1 = tf.nn.l2_normalize(normal_out_1, 2)
		tensor_mix_2 = tensor_1_transf * tensor_2_transf * tensor_mix_1
		tensor_mix_2 = tf.nn.dropout(tensor_mix_2,drop_out_rate_1)
		pooling_2=tf.nn.max_pool(tf.reshape(tensor_mix_2,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		# pdb.set_trace()
		normal_out_2 = tf.reshape(pooling_2,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_2 = tf.nn.l2_normalize(normal_out_2, 2)
		concat_output = tf.concat([normal_out_1,normal_out_2],axis = 2)
		return concat_output
	def co_att_MFH(self,question_emb, image_emb,drop_out_rate_1,drop_out_rate_mfh):
		dim_hidden_MFH = self.dim_hidden * 2
		self.conv1d_kernel_1,self.conv1d_b_1 = self.conv1d_param(self.dim_hidden,1,dim_hidden_MFH/2)
		self.conv1d_kernel_2,self.conv1d_b_2 = self.conv1d_param(dim_hidden_MFH,1,self.num_ques_glimpse_coatt/2)
		self.conv1d_kernel_3,self.conv1d_b_3 = self.conv1d_param(self.max_words_q,1,self.K/2)
		self.conv1d_kernel_4,self.conv1d_b_4 = self.conv1d_param(dim_hidden_MFH,1,dim_hidden_MFH/2)
		self.conv1d_kernel_5,self.conv1d_b_5 = self.conv1d_param(dim_hidden_MFH,1,self.num_img_glimpse_coatt/2)
		tf.add_to_collection("model_2_var", self.conv1d_kernel_1)
		tf.add_to_collection("model_2_var", self.conv1d_b_1)
		tf.add_to_collection("model_2_var", self.conv1d_kernel_2)
		tf.add_to_collection("model_2_var", self.conv1d_b_2)
		tf.add_to_collection("model_2_var", self.conv1d_kernel_4)
		tf.add_to_collection("model_2_var", self.conv1d_b_4)
		tf.add_to_collection("model_2_var", self.conv1d_kernel_5)
		tf.add_to_collection("model_2_var", self.conv1d_b_5)
		
		image_emb = tf.nn.dropout(image_emb, keep_prob=drop_out_rate_1)
		question_emb = tf.nn.dropout(question_emb, keep_prob=drop_out_rate_1)
		question_conv_1 = tf.nn.conv1d(question_emb, self.conv1d_kernel_1, stride=1,padding='SAME') + self.conv1d_b_1
		question_conv_1 = tf.nn.relu(question_conv_1)
		# question_conv_1 = tf.nn.dropout(question_conv_1, keep_prob=drop_out_rate_1)
		question_conv_2 = tf.nn.conv1d(question_conv_1, self.conv1d_kernel_2, stride=1,padding='SAME') + self.conv1d_b_2
		question_conv_2 = tf.nn.softmax(question_conv_2,dim=1)
		
		question_conv_2_1 = question_conv_2[:,:,0:1] * question_emb
		question_conv_2_2 = question_conv_2[:,:,1:2] * question_emb
		question_conv_2 = tf.concat([question_conv_2_1,question_conv_2_2],axis=2)
		question_conv_2 = tf.reduce_sum(question_conv_2, axis=1)
		question_conv_2 = tf.expand_dims(question_conv_2, axis=1)
		dim_1 = question_conv_2.shape.as_list()[2]
		dim_2 = image_emb.shape.as_list()[2]
		dim_middle = dim_hidden_MFH * 2
		dim_out = dim_hidden_MFH
		self.mfb_transf_1_1,_ = self.conv1d_param(dim_1,1,dim_middle/4)
		self.mfb_transf_1_2,_ = self.conv1d_param(dim_2,1,dim_middle/4)
		tf.add_to_collection("model_2_var", self.mfb_transf_1_1)
		tf.add_to_collection("model_2_var", self.mfb_transf_1_2)
		
		# question_conv_2 = tf.nn.dropout(question_conv_2, keep_prob=drop_out_rate_1)
		tensor_1_transf = tf.nn.conv1d(question_conv_2, self.mfb_transf_1_1, stride=1,padding='SAME')
		# image_emb = tf.nn.dropout(image_emb, keep_prob=drop_out_rate_1)
		tensor_2_transf = tf.nn.conv1d(image_emb, self.mfb_transf_1_2, stride=1,padding='SAME')
		tensor_mix_1 = tensor_1_transf * tensor_2_transf
		tensor_mix_1 = tf.nn.dropout(tensor_mix_1,drop_out_rate_mfh)
		pooling_rate = dim_middle / dim_out
		pooling_1=tf.nn.max_pool(tf.reshape(tensor_mix_1,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		normal_out_1 = tf.reshape(pooling_1,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_1 = tf.nn.l2_normalize(normal_out_1, 2)
		tensor_mix_2 = tensor_1_transf * tensor_2_transf * tensor_mix_1
		tensor_mix_2 = tf.nn.dropout(tensor_mix_2,drop_out_rate_mfh)
		pooling_2=tf.nn.max_pool(tf.reshape(tensor_mix_2,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		normal_out_2 = tf.reshape(pooling_2,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_2 = tf.nn.l2_normalize(normal_out_2, 2)
		first_mix = tf.concat([normal_out_1,normal_out_2],axis = 2)
		
		# first_mix = tf.nn.dropout(first_mix, keep_prob=drop_out_rate_1)
		question_conv_4 = tf.nn.conv1d(first_mix, self.conv1d_kernel_4, stride=1,padding='SAME') + self.conv1d_b_4
		question_conv_4 = tf.nn.relu(question_conv_4)
		# question_conv_4 = tf.nn.dropout(question_conv_4, keep_prob=drop_out_rate_1)
		question_conv_5 = tf.nn.conv1d(question_conv_4, self.conv1d_kernel_5, stride=1,padding='SAME') + self.conv1d_b_5
		# pdb.set_trace()
		question_conv_5 = tf.nn.softmax(question_conv_5,dim=1)
		self.image_att_mfh_1 = question_conv_5[:,:,0]
		self.image_att_mfh_2 = question_conv_5[:,:,1]
		image_att_feat_0 = question_conv_5[:,:,0:1] * image_emb
		image_att_feat_1 = question_conv_5[:,:,1:2] * image_emb
		image_att_feat_concat = tf.concat([image_att_feat_0,image_att_feat_1],axis = 2)
		
		dim_1 = image_att_feat_concat.shape.as_list()[2]
		dim_2 = question_conv_2.shape.as_list()[2]
		self.mfb_transf_2_1,_ = self.conv1d_param(dim_1,1,dim_middle/4)
		self.mfb_transf_2_2,_ = self.conv1d_param(dim_2,1,dim_middle/4)
		tf.add_to_collection("model_2_var", self.mfb_transf_2_1)
		tf.add_to_collection("model_2_var", self.mfb_transf_2_2)
		
		# image_att_feat_concat = tf.nn.dropout(image_att_feat_concat, keep_prob=drop_out_rate_1)
		tensor_1_transf = tf.nn.conv1d(image_att_feat_concat, self.mfb_transf_2_1, stride=1,padding='SAME')
		# question_conv_2 = tf.nn.dropout(question_conv_2, keep_prob=drop_out_rate_1)
		tensor_2_transf = tf.nn.conv1d(question_conv_2, self.mfb_transf_2_2, stride=1,padding='SAME')
		tensor_mix_1 = tensor_1_transf * tensor_2_transf
		tensor_mix_1 = tf.nn.dropout(tensor_mix_1,drop_out_rate_mfh)
		pooling_rate = dim_middle / dim_out
		pooling_1=tf.nn.max_pool(tf.reshape(tensor_mix_1,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		normal_out_1 = tf.reshape(pooling_1,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_1 = tf.nn.l2_normalize(normal_out_1, 2)
		tensor_mix_2 = tensor_1_transf * tensor_2_transf * tensor_mix_1
		tensor_mix_2 = tf.nn.dropout(tensor_mix_2,drop_out_rate_mfh)
		pooling_2=tf.nn.max_pool(tf.reshape(tensor_mix_2,shape = [self.batch_size,self.K,dim_middle/2,1]),[1,1,pooling_rate,1],[1,1,pooling_rate,1],padding='VALID')
		normal_out_2 = tf.reshape(pooling_2,shape = [self.batch_size,self.K,dim_out/2])
		normal_out_2 = tf.nn.l2_normalize(normal_out_2, 2)
		second_mix = tf.concat([normal_out_1,normal_out_2],axis = 2)
		
		pooling_2 = tf.reduce_sum(second_mix, axis=1)
		return pooling_2
		
if __name__ == "__main__":
	ac= 10


