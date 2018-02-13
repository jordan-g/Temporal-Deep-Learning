import numpy as np
import matplotlib.pyplot as plt
import time

plt.ion()

class Plotter:
	def __init__(self, title="", xlabel="", ylabel=""):
		self.figure = plt.figure()
		self.axes   = self.figure.add_subplot(1, 1, 1)

		self.axes.set_title(title)
		self.axes.set_xlabel(xlabel)
		self.axes.set_ylabel(ylabel)

		self.initial_plot_done = False

	def plot(self, y_points, labels=None, x_point=None):
		if type(y_points) not in (list, tuple):
			y_points = [y_points]
			
		if not self.initial_plot_done:
			self.x     = []
			self.ys    = [[] for i in range(len(y_points))]
			self.lines = [[] for i in range(len(y_points))]

			if x_point is None:
				x_point = len(self.ys[0])

			self.x.append(x_point)

			for i in range(len(y_points)):
				self.ys[i].append(y_points[i])

				if labels is None:
					self.lines[i], = self.axes.plot(self.x, self.ys[i])
				else:
					self.lines[i], = self.axes.plot(self.x, self.ys[i], label=labels[i])

			if labels is not None:
				self.axes.legend()

			self.initial_plot_done = True
		else:
			if x_point is None:
				x_point = len(self.ys[0])

			self.x.append(x_point)

			for i in range(len(y_points)):
				self.ys[i].append(y_points[i])

				self.lines[i].set_xdata(self.x)
				self.lines[i].set_ydata(self.ys[i])

		self.axes.relim()
		self.axes.autoscale_view()
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()
		time.sleep(1e-6)