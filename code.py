from PIL import Image
import cvxpy as cvx
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import imageio
import PIL
# Python program for random
# binary string generation


import random


# Function to create the
# random binary string
def rand_key(p):

	# Variable to store the
	# string
	key1 = ""

	# Loop to find the string
	# of desired length
	for i in range(p):
		
		# randint function to generate
		# 0, 1 randomly and converting
		# the result into str
		temp = str(random.randint(0, 1))

		# Concatenation the random 0, 1
		# to the final result
		key1 += temp
		
	return(key1)

# Driver Code
n = 7
str1 = rand_key(n)
print("Desired length random binary string is: ", str1)



value = str1
cmap = {'0': (255,255,255),
        '1': (0,0,0)}

data = [cmap[letter] for letter in value]
img = Image.new('RGB', (8, len(value)//8), "white")
img.putdata(data)
img.show()  

img.save("binary.jpg")


x_orig = imageio.imread('binary.jpg', pilmode='L') # read in grayscale
x = spimg.zoom(x_orig, -1)
ny,nx = x.shape

k = round(nx * ny * 0.5)
ri = np.random.choice(nx * ny, k, replace=False)
y = x.T.flat[ri]

psi = spfft.idct(np.identity(nx*ny), norm='ortho', axis=0)
theta = psi[ri,:] #equivalent to phi*psi


vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [theta@vx == y]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

s = np.array(vx.value).squeeze()
x_recovered = psi@s
x_recovered = x_recovered.reshape(nx, ny).T
x_recovered_final = x_recovered.astype('uint8')

imageio.imwrite('binary_recovered.jpg', x_recovered_final)
im = Image.open('binary_recovered.jpg')
im.show()
