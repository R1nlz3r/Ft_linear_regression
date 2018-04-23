import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, theta):
	plt.plot(data[:, 0], data[:, 1], 'o')
	plt.ylabel("Price")
	plt.xlabel("Km")
	plt.show()

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x

def compute_gradients(x, y, m, theta, alpha, iterations):
	for i in range(0, iterations):
		if (i == 0):
			tmp_theta = theta.copy()
		else:
			tmp_theta = np.zeros((1, 2))
		for j in range(0, m):
			tmp_theta[0, 0] += (estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j])
			tmp_theta[0, 1] += ((estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j]) * x[j])
		theta -= (tmp_theta * alpha) / m
	return theta

def main():
	data = np.loadtxt("data.csv", delimiter = ',', skiprows = 1)
	x = (data[:, 0] - min(data[:, 0])) / (max(data[:, 0]) - min(data[:, 0]))
	y = (data[:, 1] - min(data[:, 1])) / (max(data[:, 1]) - min(data[:, 1]))
	m = len(data)
	theta = np.zeros((1, 2))
	alpha = 0.3
	iterations = 1000
	theta = compute_gradients(x, y, m, theta, alpha, iterations)
	plot_data(data, theta)

if __name__ == "__main__":
	main()
