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
		self.num_output= num_output
		self.K = 36
		self.hid = dim_hidden
		self.lamb = 10e-8
		self.model_num = 2
		
		self.pre_word_embedding = pre_word_embedding
		self.num_img_glimpse_coatt = 1
		self.num_ques_glimpse_coatt = 1
		# image-embedding

		# score-embedding
		self.embed_scor_W_mfh = self.get_weights(w_shape=[self.dim_hidden, self.num_output], name="embed_scor_W_mfh", lamb=self.lamb)
		self.embed_scor_b_mfh = self.get_bias(b_shape=[self.num_output], name="embed_scor_b_mfh")

	# cnn
	def conv1d_param(self,input_tensor_length, kernel_size, feature):
		
		kernel = tf.Variable(tf.truncated_normal(shape=[kernel_size, input_tensor_length, feature], stddev=0.1), name="W")
		weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.lamb)
		tf.add_to_collection("losses", weight_decay)
		tf.add_to_collection("losses_1", weight_decay)
		tf.add_to_collection("losses_2", weight_decay)
		b = tf.Variable([feature], 
						dtype=tf.float32)
		return kernel,b
	
	def get_weights(self, name, w_shape, lamb,train = True):
		if train:
			weight = tf.Variable(tf.random_uniform(w_shape, -0.08, 0.08),name=name)
		else:
			weight = tf.random_uniform(w_shape, -0.08, 0.08)
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
	def conv1d_layer(self,input_tensor,conv1d_kernel, conv1d_b=None):
		
		feature = conv1d_kernel.shape.as_list()[2]/2
		if conv1d_b!=None:
			output_conv1d = tf.nn.conv1d(input_tensor, conv1d_kernel, 1, 'SAME') + conv1d_b
		else:
			output_conv1d = tf.nn.conv1d(input_tensor, conv1d_kernel, 1, 'SAME')
		
		conv1_half_1 = output_conv1d[:,:,0:feature]
		conv1_half_2 = output_conv1d[:,:,feature:feature*2]
		# pdb.set_trace()
		conv1_half_2 = tf.nn.softmax(conv1_half_2,dim=1)
		conv1_half_2 = conv1_half_1 * conv1_half_2
		
		output = conv1_half_2
		
		return output
		
	def forward_pass_cnn(self, tensor_x,train_val):
	
		feature_dim = self.dim_hidden/2
		conv1d_kernel_11,conv1d_b_11 = self.conv1d_param(300 + self.max_words_q,1,feature_dim)
		conv1d_kernel_12,conv1d_b_12 = self.conv1d_param(300 + self.max_words_q,3,feature_dim)
		conv1d_kernel_13,conv1d_b_13 = self.conv1d_param(300 + self.max_words_q,5,feature_dim)
		conv1d_kernel_14,conv1d_b_14 = self.conv1d_param(300 + self.max_words_q,7,feature_dim)
		
		output_0 = self.conv1d_layer(tensor_x,conv1d_kernel_11,conv1d_b_11)
		output_1 = self.conv1d_layer(tensor_x,conv1d_kernel_12,conv1d_b_12)
		output_2 = self.conv1d_layer(tensor_x,conv1d_kernel_13,conv1d_b_13)
		output_3 = self.conv1d_layer(tensor_x,conv1d_kernel_14,conv1d_b_14)
		output = tf.concat([output_0,output_1,output_2,output_3],axis=2)
		
		return output
		
	def model(self):

		image = tf.placeholder(tf.float32, [self.batch_size, self.K, self.dim_image[2]]) # b*36*2048
		label = tf.placeholder(tf.int64, [self.batch_size, ]) # b
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q]) 
		appendixes = tf.placeholder('float32', [self.batch_size, self.max_words_q,self.max_words_q], name = "appendixes")
		drop_out_rate = tf.placeholder(tf.float32)
		drop_out_rate_mfh = tf.placeholder(tf.float32)
		drop_out_rate_1 = tf.placeholder(tf.float32)
		header = tf.placeholder(tf.float32, [self.batch_size, self.max_words_q,1]) 
		footer = tf.placeholder(tf.float32, [self.batch_size, self.max_words_q,1])
		
		self.drop_out_rate = drop_out_rate
		self.drop_out_rate_mfh = drop_out_rate_mfh
		self.drop_out_rate_1 = drop_out_rate_1
		word_embeddings = []
		for i in range(self.max_words_q):
			word_emb = tf.nn.embedding_lookup(self.pre_word_embedding, question[:,i])
			# word_emb = tf.nn.dropout(word_emb, self.drop_out_rate, name = "word_emb" + str(i))
			word_embeddings.append(word_emb)

		word_embeddings = tf.transpose(word_embeddings, perm=[1, 0, 2])
		# pdb.set_trace()
		word_embeddings = tf.concat([word_embeddings,appendixes],axis=2)
		tensor_x = tf.convert_to_tensor(word_embeddings)
		output = self.forward_pass_cnn(tensor_x,'train')
		comb_emb,question_conv_2,question_conv_2b = self.co_att_MFH(output,image,header,footer)
		comb_emb = tf.nn.dropout(comb_emb, keep_prob=self.drop_out_rate)
		
		scores_emb_1 = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W_mfh, self.embed_scor_b_mfh)
		
		print("classification nums")
		
		# answer = model_att_out_0 * scores_emb_0 + model_att_out_1 * scores_emb_1
		# pdb.set_trace()
		answer = scores_emb_1
		return image, question, label,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_mfh,drop_out_rate_1,header,footer,question_conv_2,question_conv_2b

	def trainer(self):
		image, question, label,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_mfh,drop_out_rate_1,header,footer,question_conv_2,question_conv_2b = self.model()
		
		# Calculate cross entropy
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=answer)
		# Calculate loss
		loss = tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses', loss)
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		predictions = tf.argmax(answer,1)
		
		# pdb.set_trace()
		return image,question,label,loss,predictions,appendixes,answer,drop_out_rate,drop_out_rate_mfh,drop_out_rate_1,header,footer
		
	def solver(self):
		image, question, label,scores_emb_1,answer,appendixes,drop_out_rate,drop_out_rate_mfh,drop_out_rate_1,header,footer,question_conv_2,question_conv_2b = self.model()
		answer = tf.nn.softmax(answer)
		return image, question,answer,appendixes,self.image_att_mfh,drop_out_rate,drop_out_rate_mfh,drop_out_rate_1,header,footer,question_conv_2,question_conv_2b

	def tensor_sqrt(self,tensor):
		size_0 = tensor.shape.as_list()[0]
		size_1 = tensor.shape.as_list()[1]
		size_2 = tensor.shape.as_list()[2]
		tensor_one = tf.ones([size_0,size_1,size_2])
		output = tensor_one + (tensor - tensor_one)/2 - (tensor - tensor_one)*(tensor - tensor_one)/8 + (tensor - tensor_one)*(tensor - tensor_one)*(tensor - tensor_one)/16
		return output
	def UMFB(self,tensor_1,tensor_2,dim_middle,dim_out):
		
		dim_1 = tensor_1.shape.as_list()[2]
		dim_2 = tensor_2.shape.as_list()[2]
		width_1 = tensor_1.shape.as_list()[1]
		width_2 = tensor_2.shape.as_list()[1]
		if width_1 > width_2:
			width = width_1
		else:
			width = width_2
		mfb_transf_1_1,mfb_b_1_1 = self.conv1d_param(dim_1,1,dim_middle)
		mfb_transf_2_1,mfb_b_2_1 = self.conv1d_param(dim_2,1,dim_middle)
		tensor_1 = tf.nn.dropout(tensor_1, keep_prob=self.drop_out_rate_1)
		tensor_1_transf_1 = tf.nn.conv1d(tensor_1, mfb_transf_1_1, stride=1,padding='SAME')
		tensor_2 = tf.nn.dropout(tensor_2, keep_prob=self.drop_out_rate_1)
		tensor_2_transf_1 = tf.nn.conv1d(tensor_2, mfb_transf_2_1, stride=1,padding='SAME')
		tensor_mix_1 = tensor_1_transf_1 * tensor_2_transf_1
		tensor_mix_1 = tf.sign(tensor_mix_1) * (self.tensor_sqrt(tf.abs(tensor_mix_1)))
		
		
		conv1d_kernel_1,conv1d_b_1 = self.conv1d_param(dim_middle,1,dim_out)
		conv1d_kernel_2,conv1d_b_2 = self.conv1d_param(dim_out/2,1,2)
		tensor_mix_1 = tf.nn.dropout(tensor_mix_1, keep_prob=self.drop_out_rate_1)
		question_conv_1 = self.conv1d_layer(tensor_mix_1,conv1d_kernel_1,conv1d_b_1)
		question_conv_1 = tf.nn.dropout(question_conv_1, keep_prob=self.drop_out_rate_1)
		question_conv_2 = tf.nn.conv1d(question_conv_1, conv1d_kernel_2, stride=1,padding='SAME') + conv1d_b_2
		question_conv_2 = tf.tanh(question_conv_2)
		question_conv_2_1 = question_conv_2[:,:,0:1]
		question_conv_2_2 = question_conv_2[:,:,1:2]
		tensor_mix_1 += (tensor_1_transf_1 * question_conv_2_1) + (tensor_2_transf_1 * question_conv_2_2)
		
		tensor_mix_1 = tf.nn.dropout(tensor_mix_1,self.drop_out_rate_mfh)
		pooling_rate = dim_middle / dim_out
		tensor_mix_1 = tf.reshape(tensor_mix_1,shape = [self.batch_size,width,pooling_rate,dim_out])
		normal_out_1 = tf.reduce_sum(tensor_mix_1, axis=2)
		
		normal_out_1 = tf.nn.l2_normalize(normal_out_1, 2)
		concat_output = normal_out_1
		return concat_output,question_conv_2
		
	def co_att_MFH(self,question_emb, image_emb,header,footer):
		# image_emb = tf.nn.dropout(image_emb, keep_prob=self.drop_out_rate)
		dim_hidden_MFH = self.dim_hidden
		conv1d_kernel_1_1,conv1d_b_1_1 = self.conv1d_param(dim_hidden_MFH,1,dim_hidden_MFH)
		conv1d_kernel_2_1,conv1d_b_2_1 = self.conv1d_param(dim_hidden_MFH/2,1,dim_hidden_MFH)
		
		conv1d_kernel_1_2,conv1d_b_1_2 = self.conv1d_param(dim_hidden_MFH,1,dim_hidden_MFH)
		conv1d_kernel_2_2,conv1d_b_2_2 = self.conv1d_param(dim_hidden_MFH/2,1,dim_hidden_MFH)
		
		question_header = question_emb * header
		question_header = tf.nn.dropout(question_header, keep_prob=self.drop_out_rate_1)
		question_conv_1_1 = self.conv1d_layer(question_header,conv1d_kernel_1_1,conv1d_b_1_1)
		question_conv_1_1 = tf.nn.dropout(question_conv_1_1, keep_prob=self.drop_out_rate_1)
		question_conv_2_1 = tf.nn.conv1d(question_conv_1_1, conv1d_kernel_2_1, stride=1,padding='SAME') + conv1d_b_2_1
		question_conv_2_1 = tf.tanh(question_conv_2_1)
		question_header_att = question_conv_2_1 * question_header
		question_header_sum = tf.reduce_sum(question_header_att, axis=1)
		question_header_sum = tf.expand_dims(question_header_sum, axis=1)
		
		
		question_fooder = question_emb * footer
		question_fooder = tf.nn.dropout(question_fooder, keep_prob=self.drop_out_rate_1)
		question_conv_1_2 = self.conv1d_layer(question_fooder,conv1d_kernel_1_2,conv1d_b_1_2)
		question_conv_1_2 = tf.nn.dropout(question_conv_1_2, keep_prob=self.drop_out_rate_1)
		question_conv_2_2 = tf.nn.conv1d(question_conv_1_2, conv1d_kernel_2_2, stride=1,padding='SAME') + conv1d_b_2_2
		question_conv_2_2 = tf.tanh(question_conv_2_2)
		question_footer_att = question_conv_2_2 * question_fooder
		question_footer_sum = tf.reduce_sum(question_footer_att, axis=1)
		question_footer_sum = tf.expand_dims(question_footer_sum, axis=1)
		
		first_mix,question_conv_2 = self.UMFB(question_footer_sum,image_emb,dim_hidden_MFH * 5,dim_hidden_MFH)
		
		conv1d_kernel_4,conv1d_b_4 = self.conv1d_param(dim_hidden_MFH,1,dim_hidden_MFH)
		conv1d_kernel_5,conv1d_b_5 = self.conv1d_param(dim_hidden_MFH/2,1,self.num_img_glimpse_coatt)
		first_mix = tf.nn.dropout(first_mix, keep_prob=self.drop_out_rate_1)
		question_conv_4 = self.conv1d_layer(first_mix,conv1d_kernel_4,conv1d_b_4)
		question_conv_4 = tf.nn.dropout(question_conv_4, keep_prob=self.drop_out_rate_1)
		question_conv_5 = tf.nn.conv1d(question_conv_4, conv1d_kernel_5, stride=1,padding='SAME') + conv1d_b_5
		question_conv_5 = tf.nn.softmax(question_conv_5,dim=1)
		# question_conv_5 = tf.tanh(question_conv_5)
		self.image_att_mfh = question_conv_5[:,:,0]
		image_att_feat = question_conv_5[:,:,0:1] * image_emb
		# image_att_feat_sum = tf.reduce_sum(image_att_feat, axis=1)
		# image_att_feat_sum = tf.expand_dims(image_att_feat_sum, axis=1)
		
		second_mix,question_conv_2b = self.UMFB(question_header_sum,image_att_feat,dim_hidden_MFH * 5,dim_hidden_MFH)
		second_mix = tf.reduce_sum(second_mix, axis=1)
		second_mix = tf.reshape(second_mix,shape=[self.batch_size,dim_hidden_MFH])
		
		return second_mix,question_conv_2,question_conv_2b
		
		"""
		first_mix = tf.reduce_sum(first_mix, axis=1)
		return first_mix
		"""
		
if __name__ == "__main__":
	ac= 10


