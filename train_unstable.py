import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from model import Model
from utils import Timer


class Trainer():
	def __init__(self):
		self._image_size = [28, 28]
		self._batch_size = 100
		self._num_classes = 10
		self._learning_rate = 1e-3
		self._max_epoch = 100
		self._out_dir = "model_unstable"
		self._log_dir = "log_unstable"
		self._result_dir = "produce_unstable.npy"
		self._ckpt_path = "model_stable/model_88_0.9293.ckpt"
		self._train_x = np.transpose(np.load("data/test_x.npy")[:5000, :, :, :], (0, 2, 3, 1))
		self._train_y = np.load("data/test_y.npy")[:5000]
		self._test_x = np.transpose(np.load("data/test_x.npy")[5000:, :, :, :], (0, 2, 3, 1))
		self._test_y = np.load("data/test_y.npy")[5000:]
		self._num_samples = len(self._train_y)
		self._test_nums = len(self._test_y)
		self._max_iter = self._num_samples // self._batch_size
		self.network = Model(self._image_size, self._batch_size, self._num_classes)
		self.timer = Timer()

	def shuffle_data(self, images, labels):
		train_x = images.copy()
		train_y = labels.copy()
		state = np.random.get_state()
		np.random.shuffle(train_x)
		np.random.set_state(state)
		np.random.shuffle(train_y)
		return train_x, train_y

	def generate_blobs(self, x, y, start, end):
		blobs = {}
		images = x[start:end, :, :, :]
		labels = y[start:end]
		blobs["images"], blobs["labels"] = images, labels
		return blobs

	def fix_variables(self, sess):
		variables_to_restore = []
		for var in tf.compat.v1.global_variables():
			if "aux_layers" not in var.name:
				variables_to_restore.append(var)
		load_fn = slim.assign_from_checkpoint_fn(self._ckpt_path, variables_to_restore, ignore_missing_vars=True)
		load_fn(sess)
		print("Load Network: " + self._ckpt_path)

	def get_save_variables(self):
		variables_to_save = []
		for var in tf.compat.v1.global_variables():
			if "aux_layers" in var.name or "attention_layers" in var.name:
				variables_to_save.append(var)
		return variables_to_save

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			with tf.device("/cpu:0"):
				global_step = tf.Variable(0, trainable=False)
				learning_rate = tf.Variable(self._learning_rate, trainable=False)
			tf.compat.v1.set_random_seed(3)
			self.network.build_network(is_training=True, aux=True, aux_training=True)
			cross_entropy, regularizer_loss, norm, losses = self.network.add_loss()
			tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)
			tf.compat.v1.summary.scalar("regularizer_loss", regularizer_loss)
			train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy+regularizer_loss, global_step=global_step)
			merged = tf.compat.v1.summary.merge_all()
			self.saver = tf.compat.v1.train.Saver(var_list=self.get_save_variables(), max_to_keep=5)
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_dir, sess.graph)
			tf.compat.v1.global_variables_initializer().run()
			self.fix_variables(sess)
			sess.run([tf.compat.v1.assign(global_step, 0), tf.compat.v1.assign(learning_rate, self._learning_rate)])
			best_acc, epoch_list, test_list = 0, [], []
			for epoch in range(self._max_epoch):
				start, end, iter = 0, self._batch_size, 1
				train_x, train_y = self.shuffle_data(self._train_x, self._train_y)
				while iter <= self._max_iter:
					self.timer.tic()
					blobs = self.generate_blobs(train_x, train_y, start, end)
					_cross_entropy, _regularizer_loss, _, _, step, summary = self.network.train_step(sess, train_op, blobs, 
						global_step, merged)
					summary_writer.add_summary(summary, step)
					self.timer.toc()
					if iter % 25 == 0:
						print(">>>Epoch: %d\n>>>Iter: %d\n>>>Cross_entropy: %.6f\n>>>Regularizer_loss: %.6f\n>>>Speed: %.6fs\n" % (epoch+1, 
							iter, _cross_entropy, _regularizer_loss, self.timer.average_time))
					start = end
					end = start + self._batch_size
					iter += 1
				acc_mean = self.test_model(sess)
				if acc_mean > best_acc:
					best_acc = acc_mean
					if acc_mean > 0.9:
						self.snapshot(sess, epoch+1, acc_mean)
				epoch_list.append(epoch + 1)
				test_list.append(acc_mean)
			summary_writer.close()
			epoch_list, test_list = np.array(epoch_list), np.array(test_list)
			np.save(self._result_dir, np.concatenate((epoch_list[np.newaxis, ...], test_list[np.newaxis, ...]), axis=0))
			plt.plot(epoch_list, test_list, color="green")
			plt.show()

	def test_model(self, sess):
		test_start, test_end, acc_sum = 0, self._batch_size, 0.
		while test_start < self._test_nums:
			images = self._test_x[test_start:test_end, :, :, :]
			labels = self._test_y[test_start:test_end]
			_, results = self.network.test_images(sess, images)
			acc_sum += np.sum((labels==results).astype(np.float32))
			test_start = test_end
			test_end = test_start + self._batch_size
		acc_mean = acc_sum / self._test_nums
		print("Accuracy of network in test set is %.6f.\n" % (acc_mean))
		return acc_mean

	def snapshot(self, sess, epoch, acc):
		network = self.network
		file_name = os.path.join(self._out_dir, "model_%d_%.4f.ckpt" % (epoch, acc))
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: %s\n" % (file_name))


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()