import numpy as np
from matplotlib import pyplot as plt
import cv2

# Create black, gray and white image of 100x100
zeros = np.zeros((100,100))
hundreds = np.full((100, 100), 100)
twohundreds = np.full((100, 100), 200)
concatenated_array = np.hstack((zeros, hundreds, twohundreds))

# Add noise
image_with_noise = concatenated_array + 50 * (np.random.rand(100, 300)-0.5)


# Kernel size
ksize = 5
# Standard deviation (sigma)
sigma = 1

# Create a 5x5 Gaussian kernel
x, y = np.mgrid[-(ksize//2):(ksize//2)+1, -(ksize//2):(ksize//2)+1] # Calculate distance from the center
gaussian_kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) # Shape of Guassian function
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum() # Normalize to 1

print(gaussian_kernel)

# Reduce noise
smoothed_image = cv2.filter2D(image_with_noise, -1, gaussian_kernel)

plt.imshow(smoothed_image, cmap="gray")
plt.show()