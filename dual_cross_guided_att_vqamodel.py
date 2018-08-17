import csv
import pdb

import tensorflow as tf
import numpy as np
rnn_cell = tf.nn.rnn_cell
import sys

csv.field_size_limit(sys.maxsize)

class Answer_Generator():

	# cnn
	def conv1d_param(self,input_tensor_length, kernel_size, feature,b_name):
		
		kernel = tf.Variable(tf.truncated_normal(shape=[kernel_size, input_tensor_length, feature*2], stddev=0.1), name="W")
		b = tf.get_variable(b_name,
						[feature*2], 
						dtype=tf.float32, 
						initializer=tf.constant_initializer(0))
		return kernel,b

	def __init__(self,batch_size,dim_image,dim_hidden,dim_attention
					   ,max_words_q, drop_out_rate, num_output, pre_word_embedding):
		print("Initializing dual cross-guided two-layer vqa model.........")
		self.batch_size = batch_size
		self.dim_image = dim_image
		self.dim_hidden = dim_hidden
		self.dim_att = dim_attention
		self.dim_q = dim_hidden
		self.max_words_q = max_words_q
		self.drop_out_rate = drop_out_rate
		self.num_output= num_output
		self.K = 36
		self.hid = dim_attention
		self.lamb = 10e-8
		self.model_num = 2
		# question-embedding
		# self.embed_question = tf.Variable(
		#	 tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_question')
		self.pre_word_embedding = pre_word_embedding

		# image-embedding

		self.embed_image_W = self.get_weights(w_shape=[self.dim_image[2], self.dim_hidden], name="embed_image_W", lamb= self.lamb)
		self.embed_image_b =self.get_bias(b_shape=[self.dim_hidden], name="embed_image_b")

		self.embed_ques_W = self.get_weights(w_shape=[self.dim_q, self.dim_hidden], name="embed_ques_W", lamb=self.lamb)
		self.embed_ques_b = self.get_bias(b_shape=[self.dim_hidden], name="embed_ques_b")

		self.img_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="img_att_W", lamb=self.lamb)
		self.img_att_b = self.get_bias(b_shape=[1], name="img_att_b")

		# image-attention-san
		self.image_att_W_san = self.get_weights(w_shape=[self.dim_hidden, self.dim_att],name= 'image_att_W',lamb=self.lamb)
		self.image_att_b_san = self.get_bias(b_shape=[self.dim_att], name='image_att_b')

		# probability-attention
		self.prob_att_W_san = self.get_weights(w_shape=[self.dim_att, 1], name= 'prob_att_W',lamb=self.lamb)
		self.prob_att_b_san = self.get_bias(b_shape=[1], name= 'prob_att_b')

		self.ques_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="ques_att_W", lamb=self.lamb)
		self.ques_att_b = self.get_bias(b_shape=[1], name="ques_att_b")

		self.qa_W_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_clf", lamb=self.lamb)
		self.qa_b_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_img")

		self.qa_W_prime_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_prime_clf", lamb=self.lamb)
		self.qa_b_prime_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_clf")
		# score-embedding
		self.embed_scor_W = self.get_weights(w_shape=[self.dim_hidden, self.num_output], name="embed_scor_W", lamb=self.lamb)
		self.embed_scor_b = self.get_bias(b_shape=[self.num_output], name="embed_scor_b")
		
		# score-embedding
		self.embed_scor_W_san = self.get_weights(w_shape=[self.dim_hidden, self.num_output], name="embed_scor_W_san", lamb=self.lamb)
		self.embed_scor_b_san = self.get_bias(b_shape=[self.num_output], name="embed_scor_b_san")

		self.feature_dim = self.dim_hidden/2
		self.conv1d_kernel_11,self.conv1d_b_11 = self.conv1d_param(300 + self.max_words_q,1,self.feature_dim,'b_11');
		self.conv1d_kernel_12,self.conv1d_b_12 = self.conv1d_param(300 + self.max_words_q,3,self.feature_dim/2,'b_12');
		self.conv1d_kernel_13,self.conv1d_b_13 = self.conv1d_param(300 + self.max_words_q,5,self.feature_dim/2,'b_13');
		
		# self.conv1d_kernel_21,self.conv1d_b_21 = self.conv1d_param(326,1,512,'b_21');
		
		self.model_att_hidden_w = self.get_weights(w_shape=[self.dim_q, self.dim_hidden],name="model_att_hidden_w", lamb=self.lamb)
		self.model_att_hidden_b = self.get_bias(b_shape=[self.dim_hidden], name="model_att_hidden_b")

		self.model_att_w = self.get_weights(w_shape=[self.dim_hidden, self.model_num],name="model_att_w", lamb=self.lamb)
		self.model_att_b = self.get_bias(b_shape=[self.model_num], name="model_att_b")
		
	def get_weights(self, name, w_shape, lamb):
		weight = tf.Variable(tf.random_uniform(w_shape, -0.08, 0.08),name=name)
		weight_decay = tf.multiply(tf.nn.l2_loss(weight), lamb)
		tf.add_to_collection("losses", weight_decay)
		return weight

	def get_bias(self, name, b_shape):
		bias = tf.Variable(tf.random_uniform(b_shape, -0.08, 0.08),name=name)
		return bias

	#conv1d
	def conv1d_layer(self,input_tensor,conv1d_kernel, conv1d_b,feature):
		
		output_conv1d = tf.nn.conv1d(input_tensor, conv1d_kernel, 1, 'SAME') + conv1d_b
		
		conv1_half_1 = output_conv1d[:,:,0:feature]
		conv1_half_2 = output_conv1d[:,:,feature:feature*2]
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
		word_embeddings = []
		for i in range(self.max_words_q):
			word_emb = tf.nn.embedding_lookup(self.pre_word_embedding, question[:,i])
			word_emb = tf.nn.dropout(word_emb, self.drop_out_rate, name = "word_emb" + str(i))
			word_embeddings.append(word_emb)

		word_embeddings = tf.transpose(word_embeddings, perm=[1, 0, 2])
		# pdb.set_trace()
		word_embeddings = tf.concat([word_embeddings,appendixes],axis=2)
		tensor_x = tf.convert_to_tensor(word_embeddings)
		output = self.forward_pass_cnn(tensor_x,'train')
		return question,output,appendixes

	def model(self):

		image = tf.placeholder(tf.float32, [self.batch_size, self.K, self.dim_image[2]]) # b*36*2048
		label = tf.placeholder(tf.int64, [self.batch_size, ]) # b
		
		# drop_out = tf.placeholder(tf.float32)
		"""
		question_feat, _ = self.rnn_model.bi_gru_question(self.batch_size, pre_word_embedding=self.pre_word_embedding,
												  inputs=question, time_step=self.max_words_q,
												  layer_num=1, hidden_size=self.rnn_size)   # b*26*1024
		"""
		question,output,appendixes = self.pre_model()
		
		image_feat = tf.nn.l2_normalize(image, -1)
		# embedding
		image_emb = tf.reshape(image_feat, [-1, self.dim_image[2]])  # (b x m) x d
		image_emb = tf.nn.xw_plus_b(image_emb,self.embed_image_W, self.embed_image_b)
		image_emb_2d = tf.nn.dropout(image_emb, keep_prob=self.drop_out_rate)
		image_emb = tf.tanh(tf.reshape(image_emb_2d, shape=[self.batch_size, self.K, self.dim_hidden])) # (b*6*6)*2048

		ques_emb = tf.reshape(output, [-1, self.dim_q])
		ques_emb = tf.nn.xw_plus_b(ques_emb, self.embed_ques_W, self.embed_ques_b)
		ques_emb = tf.nn.dropout(ques_emb, keep_prob=self.drop_out_rate)
		ques_emb = tf.tanh(tf.reshape(ques_emb, shape=[self.batch_size,-1, self.dim_hidden]))  # (b*26)*1024

		# first layer attention
		image_emb_att = tf.nn.xw_plus_b(tf.reshape(image_emb, shape=[-1,self.dim_hidden]), self.img_att_W, self.img_att_b)
		self.image_emb_prob = tf.nn.softmax(tf.reshape(image_emb_att, shape=[self.batch_size, -1]))
		image_emb_prob = self.image_emb_prob
		img_memory = tf.reduce_sum(tf.expand_dims(image_emb_prob,2)*image_emb, axis=1)
		# reduce
		ques_emb_att = tf.nn.xw_plus_b(tf.reshape(ques_emb, shape=[-1,self.dim_hidden]), self.ques_att_W, self.ques_att_b)
		ques_emb_prob = tf.nn.softmax(tf.reshape(ques_emb_att, shape=[self.batch_size, -1]))
		ques_memory = tf.reduce_sum(tf.expand_dims(ques_emb_prob, 2) * ques_emb, axis=1)
		# ques * image
		memory = img_memory*ques_memory  # b*1024
		# # attention models
		with tf.variable_scope("att1"):
			# vis_comb1 512   ques_comb1 1024
			self.vis_att_prob1, vis_comb1 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb)
			self.ques_att_prob1, ques_comb1 =self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb)
			img_memory = img_memory + vis_comb1
			ques_memory = ques_memory + ques_comb1
			memory = memory + img_memory*ques_memory
		with tf.variable_scope("att2"):
			self.vis_att_prob2, vis_comb2 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb)
			self.ques_att_prob2, ques_comb2 = self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb)
			img_memory = img_memory + vis_comb2
			ques_memory = ques_memory + ques_comb2
			memory = memory + img_memory * ques_memory

		s_head = self.gated_tanh(memory, self.qa_W_clf, self.qa_b_clf, self.qa_W_prime_clf, self.qa_b_prime_clf)
		# s_head = memory * memory * memory
		"""
		begin
		"""
		
		
		question_emb = tf.reduce_max(output, 1)
		#attention models
		with tf.variable_scope("san_att1"):
			san_prob_att1, comb_emb = self.attention_san(question_emb, image_emb_2d)
		with tf.variable_scope("san_att2"):
			san_prob_att2, comb_emb = self.attention_san(comb_emb, image_emb_2d)

		# head = model_att_out_0 * s_head + model_att_out_1 * comb_emb
		s_head = tf.nn.dropout(s_head, keep_prob=self.drop_out_rate)
		comb_emb = tf.nn.dropout(comb_emb, keep_prob=self.drop_out_rate)
		"""
		end
		"""
		
		scores_emb_0 = tf.nn.xw_plus_b(s_head, self.embed_scor_W, self.embed_scor_b)
		scores_emb_1 = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W_san, self.embed_scor_b_san)
		"""
		begin
		"""
		# scores_emb = tf.tanh(scores_emb)
		"""
		end
		"""
		print("classification nums")
		# print(scores_emb)
		
		return image, question, label, scores_emb_0,scores_emb_1,appendixes

	def trainer(self):
		image, question, label, scores_emb_0,scores_emb_1,appendixes = self.model()
		scores_emb = tf.concat([scores_emb_0,scores_emb_1], axis=0)
		label = tf.concat([label,label], axis=0)
		# Calculate cross entropy
		# pdb.set_trace()
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=scores_emb)
		# Calculate loss
		loss = tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses', loss)
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		predictions = tf.argmax(scores_emb,1)
		
		# pdb.set_trace()
		return image,question,label,loss,predictions,appendixes

	"""
	def solver(self):
		image, question, label, scores_emb_0,scores_emb_1,appendixes = self.model()
		# answer_prob = tf.nn.softmax(scores_emb)
		answer_prob_0 = scores_emb_0
		answer_prob_1 = scores_emb_1
		return image, question, answer_prob_0,answer_prob_1,appendixes
	"""

	def model_r(self):
		label_r = tf.placeholder(tf.float32, [self.batch_size, self.model_num])
		"""
		predictions_0 = tf.placeholder(tf.int64, [self.batch_size, ]) # b
		predictions_1 = tf.placeholder(tf.int64, [self.batch_size, ]) # b
		"""
		question_r,cnn_output_r,appendixes_r = self.pre_model()
		
		question_emb = tf.reduce_max(cnn_output_r, 1)
		question_emb = tf.nn.dropout(question_emb, keep_prob=self.drop_out_rate)
		# pdb.set_trace()
		model_att_hidden = tf.nn.xw_plus_b(question_emb, self.model_att_hidden_w, self.model_att_hidden_b)
		model_att_hidden =  tf.tanh(model_att_hidden)
		model_att_out = tf.nn.xw_plus_b(model_att_hidden, self.model_att_w, self.model_att_b)
		# model_att_out = tf.nn.softmax(model_att_out)
		# model_att_out = tf.sigmoid(model_att_out)
		# model_att_out = tf.tanh(model_att_out)
		
		return label_r,question_r,appendixes_r,model_att_out
	def trainer_r(self):
		label_r,question_r,appendixes_r,model_att_out = self.model_r()
		
		# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_r, logits=model_att_out)
		# pdb.set_trace()
		cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_r, logits=model_att_out)
		# Calculate loss
		loss_r = tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses_r', loss_r)
		loss_r = tf.add_n(tf.get_collection('losses_r'), name='total_loss_r')
		predictions_r = tf.argmax(model_att_out,1)
		return question_r,label_r,appendixes_r,loss_r,predictions_r,model_att_out,cross_entropy
	def solver_r(self):
		label_r,question_r,appendixes_r,model_att_out = self.model_r()
		image, question,label,answer_prob_0,answer_prob_1,appendixes = self.model()
		model_att_out = tf.sigmoid(model_att_out)
		self.model_att = model_att_out
		max_index = tf.argmax(model_att_out,1)
		max_index = tf.expand_dims(max_index, 1)
		model_att_out_1 = tf.tile(model_att_out[:,0:1], tf.constant([1, self.num_output]))
		model_att_one = np.ones((self.batch_size, self.num_output))
		model_att_out_0 = model_att_one - model_att_out_1
		# answer = model_att_out_0 * answer_prob_0 + model_att_out_1 * answer_prob_1
		answer = answer_prob_0
		return image, question,question_r, answer,appendixes,appendixes_r, self.image_emb_prob, self.vis_att_prob1, self.vis_att_prob2, self.ques_att_prob1, self.ques_att_prob2,self.model_att
	def gated_tanh(self, concated, w1, b1, w2, b2):
		y_tilde = tf.tanh(tf.nn.xw_plus_b(concated, w1, b1))
		g = tf.sigmoid(tf.nn.xw_plus_b(concated, w2, b2))
		y = tf.multiply(y_tilde, g)
		return y


	def tanh_vis_attention(self, question_emb, image_emb):
		# Attention weight
		# question_emb b*1024  image_emb b*K*2048
		# question-attention
		# probability-attention
		gt_W_img_att = self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_img_att", lamb=self.lamb)
		gt_b_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_img_att")
		gt_W_prime_img_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_img_att", lamb=self.lamb)
		gt_b_prime_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_img_att")
		prob_image_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_image_att_W", lamb=self.lamb)
		prob_image_att_b = self.get_bias(b_shape=[1], name="prob_image_att_b")
		# pdb.set_trace()
		qenc_reshape = tf.tile(tf.expand_dims(question_emb, 1), multiples=[1, self.K, 1])  # b * k * 1024
		concated = tf.concat([image_emb, qenc_reshape], axis=2)  # b * m * (image_dim + ques_dim)
		concated = tf.reshape(concated, shape=[self.batch_size * self.K, -1])
		concated = self.gated_tanh(concated, gt_W_img_att, gt_b_img_att, gt_W_prime_img_att, gt_b_prime_img_att)  # (b * m) * hid
		att_map = tf.nn.xw_plus_b(concated, prob_image_att_W, prob_image_att_b)  # b*m*1
		att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.K]))
		v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * image_emb, axis=1)
		return att_prob, v_head

	def tanh_ques_attention(self, question_emb, image_emb):
		# Attention weight
		gt_W_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_ques_att", lamb=self.lamb)
		gt_b_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_ques_att")

		gt_W_prime_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_ques_att", lamb=self.lamb)
		gt_b_prime_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_ques_att")

		prob_ques_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_ques_att_W", lamb=self.lamb)
		prob_ques_att_b = self.get_bias(b_shape=[1], name="prob_ques_att_b")

		img_reshape = tf.tile(tf.expand_dims(image_emb, 1), multiples=[1, self.max_words_q, 1])  # b * 26 * 1024
		concated = tf.concat([question_emb, img_reshape], axis=2)  # b * 26 * (image_dim + ques_dim)
		concated = tf.reshape(concated, shape=[self.batch_size * self.max_words_q, -1])
		concated = self.gated_tanh(concated, gt_W_ques_att, gt_b_ques_att, gt_W_prime_ques_att, gt_b_prime_ques_att)  # (b * m) * hid
		att_map = tf.nn.xw_plus_b(concated, prob_ques_att_W, prob_ques_att_b)  # b*m*1
		att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.max_words_q]))
		v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * question_emb, axis=1)
		return att_prob, v_head

	def attention_san(self, question_emb, image_emb):
		
		question_att = tf.expand_dims(question_emb, 1) # b x 1 x d
		question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1]*self.dim_att/self.dim_hidden, 1])) # b x m x d
		question_att = tf.reshape(question_att, [-1, self.dim_att]) # (b x m) x d
		# question_att = tf.tanh(tf.nn.xw_plus_b(question_att, ques_att_W, ques_att_b)) # (b x m) x k
		
		image_att = tf.tanh(tf.nn.xw_plus_b(image_emb, self.image_att_W_san, self.image_att_b_san)) # (b x m) x k
		
		output_att = tf.tanh(image_att + question_att) # (b x m) x k
		output_att = tf.nn.dropout(output_att,self.drop_out_rate)
		prob_att = tf.nn.xw_plus_b(output_att, self.prob_att_W_san, self.prob_att_b_san) # (b x m) x 1
		prob_att = tf.reshape(prob_att, [self.batch_size, self.dim_image[0] * self.dim_image[1]]) # b x m
		prob_att = tf.nn.softmax(prob_att)

		image_att = []
		image_emb = tf.reshape(image_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1], self.dim_hidden]) # b x m x d
		for b in range(self.batch_size):
			image_att.append(tf.matmul(tf.expand_dims(prob_att[b,:],0), image_emb[b,:,:]))

		image_att = tf.stack(image_att)
		image_att = tf.reduce_sum(image_att, 1)
		comb_emb = tf.add(image_att, question_emb)
		
		return prob_att, comb_emb
if __name__ == "__main__":
	ac= 10


