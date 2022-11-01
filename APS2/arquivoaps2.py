#Disciplina: Métodos Numéricos Computacionais - 145R
#Aluno: Bryan de Oliveira Nery - 2020100754
#Questão APS2

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
import cv2
from skimage.color import rgb2gray
from numpy.linalg import svd

origem = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTb3tNPVY4Nfz4ciprlKFkGOWbDPL9l1Vgfpw&usqp=CAU"
image = io.imread(origem) 
plt.imshow(image)
# convert to grayscale
gray_img = rgb2gray(image)

# calculate the SVD and plot the image
U,S,V_T = svd(gray_img, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(3, 2, figsize=(8, 20))

curr_fig=0
print(image.shape)
kv1 = np.ceil(0.6*240)
kv2 = np.ceil(0.7*240)
kv3 = np.ceil(0.8*240)
for r in [int(kv1), int(kv2), int(kv3)]:
  img_approx=U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
  ax[curr_fig][0].imshow(256-img_approx)
  ax[curr_fig][0].set_title("k = "+str(r))
  ax[curr_fig,0].axis('off')
  ax[curr_fig][1].set_title("Original Image")
  ax[curr_fig][1].imshow(gray_img)
  ax[curr_fig,1].axis('off')
  curr_fig +=1
plt.show()
