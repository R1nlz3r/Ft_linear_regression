import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, x, y):
	plt.plot(data[:, 0], data[:, 1], 'o')
	plt.plot(x, y)
	plt.ylabel("Price")
	plt.xlabel("Km")
	plt.show()

def standardize(x):
	return (x - np.mean(x)) / np.std(x)

def destandardize(x, x_ref):
	return x * np.std(x_ref) + np.mean(x_ref)

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
	data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)
	x = standardize(data[:, 0])
	y = standardize(data[:, 1])
	m = len(data)
	theta = np.zeros((1, 2))
	alpha = 0.3
	iterations = 200
	theta = compute_gradients(x, y, m, theta, alpha, iterations)
	y = estimate_price(theta[0, 0], theta[0, 1], x)
	x = destandardize(x, data[:, 0])
	y = destandardize(y, data[:, 1])
	plot_data(data, x, y)

if __name__ == "__main__":
	main()
