import time
import cv2

def corner_detector(image,threshold): #Returns corner indexes
    window_size = 3
    r = window_size // 2 # It is used for storing sum of products in center point.
    corner_points = [] # Holds corner points's x and y values
    min_values = [] # Holds minimum eigenvalues are higher than threshold
    for i in range(420,700): # It observe rows of image where all shapes are located.
        for j in range(1,image.shape[1]-window_size):
            tensor = [[0,0],[0,0]] # It holds product of sum of gradients. It refers "Image Structure Tensor" in lecture slides.
            for k in range(window_size): # It is for window based operation
                for l in range(window_size):
                    tensor[0][0] += ((int(image[i+k+1][j+l])-int(image[i+k-1][j+l]))/2)**2
                    tensor[0][1] += ((int(image[i+k+1][j+l])-int(image[i+k-1][j+l]))/2) * ((int(image[i+k][j+l+1])-int(image[i+k][j+l-1]))/2)
                    tensor[1][0] += ((int(image[i+k+1][j+l])-int(image[i+k-1][j+l]))/2) * ((int(image[i+k][j+l+1])-int(image[i+k][j+l-1]))/2)
                    tensor[1][1] += ((int(image[i+k][j+l+1])-int(image[i+k][j+l-1]))/2)**2
            min_value = 0.5 * ((tensor[0][0]+tensor[1][1])-((tensor[0][0]-tensor[1][1])**2 + 4*(tensor[0][1])**2)**0.5) #Minimum eigenvalue calculation according to lecture slide "lambda2 = 0.5*( (s11+s22)-((s11-s22).^2+4*s12.^2).^0.5 )"
            if min_value > threshold: #Add corner if minimum eigenvalue is greater or equal to threshold value
                min_values.append(min_value)
                corner_points.append([]) #Add a list to hold x and y values
                corner_points[len(corner_points) - 1].append(i + r) #Add x value to related corner
                corner_points[len(corner_points) - 1].append(j + r) #Add y value to related corner
    return min_values, corner_points

image = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE) #Read screen shot as greyscale image
mins, corners = corner_detector(image,8000) #Detects corners with minimum eigenvalue approach

#Below section draws corner points on screenshot image.
im = cv2.imread("test.png")
for corner in corners:
    x = corner[0]
    y = corner[1]
    cv2.circle(im,(y,x),3,(255,0,0),-1)
cv2.imshow("corners",im)
cv2.waitKey(0)

