import matplotlib.pyplot as plt
import numpy as np



def save_theta(theta, x, y):
	a = (y[0] - y[1]) / (x[0] - x[1])
	b = a * x[0] * -1 + y[0]
	theta = [b, a]
	np.savetxt("theta.csv", theta, delimiter = ',');


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


#	Adjusts the weights 'theta' a number 'iterations' of times at an 'alpha' learning rate.
#	The data 'x' and 'y' should be standardized and the data length passed as 'm'.

def compute_gradients(x, y, m, theta, alpha, iterations):
	for i in range(0, iterations):
		tmp_theta = np.zeros((1, 2))
		for j in range(0, m):
			tmp_theta[0, 0] += (estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j])
			tmp_theta[0, 1] += ((estimate_price(theta[0, 0], theta[0, 1], x[j]) - y[j]) * x[j])
		theta -= (tmp_theta * alpha) / m
	return theta


#	Computes the linear regression weights 'theta' of the 'data.csv' dataset.
#	Saves the weights in 'theta.csv' for other program purposes.
#	Plots the data along with the estimates.

def main():
	data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)

	if (len(data) < 2):
		exit()

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

	save_theta(theta, x, y)

	plot_data(data, x, y)

if __name__ == "__main__":
	main()
