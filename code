from PIL import Image
import cvxpy as cvx
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import imageio
import PIL

value = "10101010101010101001011110010101001010100101010100001010101011010111011110101010010100101010101111100100"

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
