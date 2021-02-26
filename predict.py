import numpy as np

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x


#	Loads the computed weights of the linear regression
#	Takes a number input and prints the prediction based on the weights

def main():
	theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
	try:
		x = np.longdouble(raw_input("Enter a number: "))
	except:
		print ("Error")
		exit()
	print estimate_price(theta[0], theta[1], x)

if __name__ == "__main__":
	main()
