import numpy as np
import time


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff


def calc_mean(hid, labels, num_classes):
	out = np.zeros_like(hid, np.float32)
	for i in range(num_classes):
		index = np.where(labels==i)[0]
		if len(index) != 0:
			extract = np.transpose(hid[index, :, :, :].copy(), (3, 0, 1, 2))
			extract = np.reshape(extract, (np.shape(extract)[0], -1))
			out[index, :, :, :] = np.mean(extract, axis=1)
	return out