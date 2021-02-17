import cv2
import numpy as np

def reflect(kernel): #kernel = np.flipud(np.fliplr(kernel)) can be used
    reflected = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            reflected[i][j] = kernel[kernel.shape[0]-1-i][kernel.shape[1]-1-j]
    return reflected

def edge_magnitude(image1, image2): #Calculates (Rx^2 + Ry^2)**0.5 for all gradient of pixels
    magnitudes = np.zeros(image2.shape)
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            magnitudes[i][j] = (image1[i][j]**2 + image2[i][j]**2)**(1/2) #Calculation of (Rx^2 + Ry^2)**0.5
    return magnitudes

def sobel(image,kernel): #sobel filter
    filtered = np.zeros(image.shape) #Initialize filtered image with zeros
    mask = reflect(kernel) #get reflected kernel
    mask_h = mask.shape[0] #height of mask
    mask_w = mask.shape[1] #width of mask

    img_h = image.shape[0] #height of image
    img_w = image.shape[1] #width of image

    x = mask_h // 2
    y = mask_w // 2
    for i in range(img_h-mask_h+1): #convolution of image and mask
        for j in range(img_w-mask_w+1):
            result = 0 #hold summations for every step
            for k in range(mask_h): #Calculates sum of products of kernel and image pixel values
                for l in range(mask_w):
                    result = result + image[i+k][j+l] * mask[k][l]
            filtered[i+x][j+y] = result #Value assigned to center pixel
    return filtered

image = cv2.imread("test2.png",cv2.IMREAD_GRAYSCALE) #Read screen shot as greyscale image
kernel1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #Horizontal edge detection mask
kernel2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Vertical edge detection mask
result1 = sobel(image,kernel1) #Horizontal edge detection
result2 = sobel(image,kernel2) #Vertical edge detection
magnitudes = edge_magnitude(result1,result2) #Image gradient magnitudes
cv2.imshow("as",magnitudes) #Shows edges on "All Shapes" page
cv2.waitKey(0)