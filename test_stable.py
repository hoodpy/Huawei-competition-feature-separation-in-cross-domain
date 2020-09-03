import tensorflow as tf
import numpy as np
from model import Model


def calc_acc(confusion_matrix):
	h_dim, w_dim, true_num = np.shape(confusion_matrix)[0], np.shape(confusion_matrix)[1], 0.
	for i in range(h_dim):
		true_num += confusion_matrix[i, i]
	return true_num / np.sum(confusion_matrix)


ckpt_path = "model_stable/model_88_0.9293.ckpt"
test_x = np.transpose(np.load("data/test_x.npy"), (0, 2, 3, 1))
test_y = np.load("data/test_y.npy")
batch_size, test_nums = 100, len(test_y)
network = Model(image_size=[28, 28], batch_size=batch_size, num_classes=10)

if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	with tf.compat.v1.Session(config=config) as sess:
		network.build_network(is_training=False, aux=False, aux_training=False)
		tf.compat.v1.global_variables_initializer().run()
		saver = tf.compat.v1.train.Saver()
		saver.restore(sess, ckpt_path)
		test_start, test_end, confusion_matrix = 0, batch_size, np.zeros((10, 10), np.int32)
		while test_start < test_nums:
			images = test_x[test_start:test_end, :, :, :]
			labels = test_y[test_start:test_end]
			_, results = network.test_images(sess, images)
			for i in range(len(labels)):
				confusion_matrix[int(labels[i]), results[i]] += 1
			test_start = test_end
			test_end = test_start + batch_size
		print("confusion_matrix: \n" + str(confusion_matrix))
		print("Accuracy of network in test set is %.6f.\n" % (calc_acc(confusion_matrix)))