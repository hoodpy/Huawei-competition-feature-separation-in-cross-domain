import tensorflow as tf
from utils import calc_mean


class Model():
	def __init__(self, image_size, batch_size, num_classes):
		self._image_size = image_size
		self._batch_size = batch_size
		self._num_classes = num_classes
		self._regularizer_l2 = tf.contrib.layers.l2_regularizer(1e-4)

	def conv2d(self, input, num_outputs, is_training, scope, kernel_size=3, stride=1, padding="SAME", bn=False, relu=True):
		with tf.compat.v1.variable_scope(scope):
			num_inputs = input.get_shape()[-1]
			weights = tf.compat.v1.get_variable("weights", [kernel_size, kernel_size, num_inputs, num_outputs], 
				initializer=tf.random_normal_initializer(stddev=0.1), regularizer=self._regularizer_l2, trainable=is_training)
			conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding=padding, name="conv2d_op")
			if bn:
				conv = tf.layers.batch_normalization(conv, training=is_training, name="bn_op")
			if relu:
				conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def incep_A(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = self.conv2d(inputs, num_outputs=input_channals//4, is_training=is_training, scope="block1", kernel_size=1)
			block2 = self.conv2d(inputs, num_outputs=input_channals//4, is_training=is_training, scope="block2", kernel_size=3)
			block3 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block3_a", kernel_size=3)
			block3 = self.conv2d(block3, num_outputs=input_channals//4, is_training=is_training, scope="block3_b", kernel_size=3)
			block4 = tf.nn.avg_pool2d(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name="block4_a")
			block4 = self.conv2d(block4, num_outputs=input_channals//4, is_training=is_training, scope="block4_b", kernel_size=1)
			block_out = tf.concat([block1, block2, block3, block4], 3)
			return block_out

	def incep_B(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = self.conv2d(inputs, num_outputs=input_channals//3, is_training=is_training, scope="block1", kernel_size=1)
			block2 = self.conv2d(inputs, num_outputs=input_channals//3+1, is_training=is_training, scope="block2", kernel_size=3)
			block3 = tf.nn.avg_pool2d(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name="block3_a")
			block3 = self.conv2d(block3, num_outputs=input_channals//3, is_training=is_training, scope="block3_b", kernel_size=1)
			block_out = tf.concat([block1, block2, block3], 3)
			return block_out

	def incep_C(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = self.conv2d(inputs, num_outputs=input_channals//3, is_training=is_training, scope="block1", kernel_size=1)
			block2 = self.conv2d(inputs, num_outputs=input_channals//3+2, is_training=is_training, scope="block2", kernel_size=3)
			block3 = tf.nn.max_pool2d(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name="block3_a")
			block3 = self.conv2d(block3, num_outputs=input_channals//3, is_training=is_training, scope="block3_b", kernel_size=1)
			block_out = tf.concat([block1, block2, block3], 3)
			return block_out

	def incep_res_A(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = tf.identity(inputs, name="block1")
			block2 = self.conv2d(inputs, num_outputs=input_channals, is_training=is_training, scope="block2", kernel_size=1)
			block3 = self.conv2d(inputs, num_outputs=input_channals, is_training=is_training, scope="block3", kernel_size=3)
			block4 = self.conv2d(inputs, num_outputs=input_channals, is_training=is_training, scope="block4_a", kernel_size=3)
			block4 = self.conv2d(block4, num_outputs=input_channals, is_training=is_training, scope="block4_b", kernel_size=3)
			block_out = tf.math.add_n([block1, block2, block3, block4])
		return block_out

	def incep_res_B(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = tf.identity(inputs, name="block1")
			block2 = self.conv2d(inputs, num_outputs=input_channals, is_training=is_training, scope="block2", kernel_size=1)
			block3 = self.conv2d(inputs, num_outputs=input_channals, is_training=is_training, scope="block3", kernel_size=3)
			block_out = tf.math.add_n([block1, block2, block3])
		return block_out

	def reduction_A(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = tf.nn.max_pool2d(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="block1")
			block2 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block2", stride=2)
			block3 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block3_a")
			block3 = self.conv2d(block3, num_outputs=input_channals//2, is_training=is_training, scope="block3_b", stride=2)
			block_out = tf.concat([block1, block2, block3], 3)
		return block_out

	def reduction_B(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = tf.nn.max_pool2d(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="block1")
			block2 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block2", kernel_size=1, stride=2)
			block3 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block3", kernel_size=3, stride=2)
			block_out = tf.concat([block1, block2, block3], 3)
		return block_out

	def reduction_C(self, inputs, is_training, scope):
		with tf.compat.v1.variable_scope(scope):
			input_channals = inputs.get_shape()[-1]
			block1 = tf.identity(inputs, name="block1")
			block2 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block2", kernel_size=1)
			block3 = self.conv2d(inputs, num_outputs=input_channals//2, is_training=is_training, scope="block3", kernel_size=3)
			block_out = tf.nn.avg_pool2d(tf.concat([block1,block2,block3],3), ksize=[1,7,7,1], strides=[1,1,1,1], padding="VALID")
		return block_out

	def aux_module(self, is_training, scope="aux_layers"):
		with tf.compat.v1.variable_scope(scope):
			conv0_0 = self.conv2d(self._images_input, num_outputs=64, is_training=is_training, scope="conv0_0")
			conv0_1 = self.conv2d(conv0_0, num_outputs=64, is_training=is_training, scope="conv0_1")
			pool_0 = tf.nn.max_pool2d(conv0_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool_0")

			conv1_0 = self.conv2d(pool_0, num_outputs=128, is_training=is_training, scope="conv1_0")
			conv1_1 = self.conv2d(conv1_0, num_outputs=128, is_training=is_training, scope="conv1_1")
			pool_1 = tf.nn.max_pool2d(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool_1")

			conv2_0 = self.conv2d(pool_1, num_outputs=256, is_training=is_training, scope="conv2_0")
			conv2_1 = self.conv2d(conv2_0, num_outputs=256, is_training=is_training, scope="conv2_1")
			conv2_2 = self.conv2d(conv2_1, num_outputs=256, is_training=is_training, scope="conv2_2")
			pool_2 = tf.nn.avg_pool2d(conv2_2, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool_2")
			pool_2 = tf.squeeze(pool_2, axis=[1, 2], name="squeeze_op")
		return pool_2

	def attention(self, is_training, scope="attention_layers"):
		with tf.compat.v1.variable_scope(scope):
			weights = tf.compat.v1.get_variable("weights", [2], initializer=tf.random_uniform_initializer(), trainable=is_training)
			attention_weights = tf.nn.softmax(weights, name="softmax_op")
		return attention_weights

	def build_network(self, is_training, aux, aux_training):
		mark1 = True if is_training and not aux else False
		mark2 = True if is_training and aux_training else False

		with tf.device("/cpu:0"):
			self._images_input = tf.compat.v1.placeholder(tf.float32, [self._batch_size, self._image_size[0], self._image_size[1], 3], 
				name="images_input")
			self._labels_input = tf.compat.v1.placeholder(tf.int32, [self._batch_size], name="labels_input")

		with tf.compat.v1.variable_scope("inference_layers"):
			hidden = self.conv2d(self._images_input, num_outputs=32, is_training=mark1, scope="conv0_1", kernel_size=5, relu=False)
			self._hidden = hidden
			conv0_1 = tf.nn.relu(hidden, name="conv0_1_relu")

			conv1_0 = self.incep_A(conv0_1, is_training=mark1, scope="incep_A")
			conv1_1 = self.incep_res_A(conv1_0, is_training=mark1, scope="incep_res_A1")
			conv1_2 = self.incep_res_A(conv1_1, is_training=mark1, scope="incep_res_A2")
			conv1_3 = self.reduction_A(conv1_2, is_training=mark1, scope="reduction_A")

			conv2_0 = self.incep_B(conv1_3, is_training=mark1, scope="incep_B")
			conv2_1 = self.incep_res_B(conv2_0, is_training=mark1, scope="incep_res_B1")
			conv2_2 = self.incep_res_B(conv2_1, is_training=mark1, scope="incep_res_B2")
			conv2_3 = self.incep_res_B(conv2_2, is_training=mark1, scope="incep_res_B3")
			conv2_4 = self.incep_res_B(conv2_3, is_training=mark1, scope="incep_res_B4")
			conv2_5 = self.incep_res_B(conv2_4, is_training=mark1, scope="incep_res_B5")
			conv2_6 = self.reduction_B(conv2_5, is_training=mark1, scope="reduction_B")

			conv3_0 = self.incep_C(conv2_6, is_training=mark1, scope="incep_C")
			conv3_1 = self.incep_res_B(conv3_0, is_training=mark1, scope="incep_res_C1")
			conv3_2 = self.incep_res_B(conv3_1, is_training=mark1, scope="incep_res_C2")
			conv3_3 = self.reduction_C(conv3_2, is_training=mark1, scope="reduction_C")
			conv3_3 = tf.squeeze(conv3_3, axis=[1, 2], name="squeeze_op")

		aux_out = self.aux_module(is_training=mark2) if aux else tf.zeros_like(conv3_3, tf.float32, name="aux_layers")
		attention_weights = self.attention(is_training=mark2) if aux else tf.constant([1.0, 0.0], tf.float32, name="attention_layers")

		with tf.compat.v1.variable_scope("classifier_layers"):
			weights0 = tf.compat.v1.get_variable("weights0", [256, 128], initializer=tf.random_normal_initializer(stddev=0.1), 
				regularizer=self._regularizer_l2, trainable=mark1)
			weights1 = tf.compat.v1.get_variable("weights1", [128, self._num_classes], regularizer=self._regularizer_l2, 
				initializer=tf.random_normal_initializer(stddev=0.1), trainable=mark1)

			integ_feature = tf.math.add(tf.math.multiply(conv3_3, attention_weights[0]), tf.math.multiply(aux_out, attention_weights[1]))
			fc = tf.nn.relu(tf.linalg.matmul(integ_feature, weights0, name="fc"), name="fc_relu")

			logits = tf.linalg.matmul(fc, weights1, name="logits")
			logits_softmax = tf.nn.softmax(logits, axis=1, name="logits_softmax")
			logits_results = tf.argmax(logits, dimension=1, name="logits_results")

			self._logits = logits
			self._logits_softmax = logits_softmax
			self._logits_results = logits_results

	def correlation_reg(self):
		mean = tf.py_func(calc_mean, [self._hidden, self._labels_input, self._num_classes], tf.float32)
		norm = tf.reduce_sum(tf.math.square(tf.math.subtract(self._hidden, mean)))
		return norm

	def add_loss(self):
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels_input))
		regularizer_loss = tf.math.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
		norm = self.correlation_reg()
		losses = cross_entropy + regularizer_loss + norm
		self._cross_entropy = cross_entropy
		self._regularizer_loss = regularizer_loss
		self._norm = norm
		self._losses = losses
		return cross_entropy, regularizer_loss, norm, losses

	def train_step(self, sess, train_op, blobs, global_step, merged):
		feed_dict={self._images_input: blobs["images"], self._labels_input: blobs["labels"]}
		_, cross_entropy, regularizer_loss, norm, losses, step, summary = sess.run([train_op, self._cross_entropy, self._regularizer_loss, 
			self._norm, self._losses, global_step, merged], feed_dict=feed_dict)
		return cross_entropy, regularizer_loss, norm, losses, step, summary

	def test_images(self, sess, images):
		probabilities, results = sess.run([self._logits_softmax, self._logits_results], feed_dict={self._images_input: images})
		return probabilities, results