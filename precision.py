import numpy as np

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x


#	Applied the Coefficient of determination formula:
#	https://fr.wikipedia.org/wiki/Coefficient_de_d%C3%A9termination
#	The range is 0 to 1.

def compute_score(x, y, m, theta):
	estimate_prices = estimate_price(theta[0], theta[1], x)
	estimate_differences = y - estimate_prices
	sum_square_explained = np.sum(np.power(y - estimate_prices, 2))

	y_mean = np.mean(y)
	sum_square_total = np.sum(np.power(y - y_mean, 2))
	print(1 - (sum_square_explained / sum_square_total))


def main():
	data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)
	theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')

	if (len(data) < 2):
		exit()

	x = data[:, 0]
	y = data[:, 1]
	m = len(data)

	compute_score(x, y, m, theta)

if __name__ == "__main__":
	main()
