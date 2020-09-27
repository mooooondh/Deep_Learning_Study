# mnist01_imgshow.py

import sys, os
sys.path.append(os.pardir)
import numpy as np
from book_ex.dataset.mnist import load_mnist
from PIL import Image

def img_show(img):  # img show function
    pil_image= Image.fromarray(np.uint8(img))
    pil_image.show()

(X_train, y_train), (X_test, y_test)= load_mnist(flatten= True, normalize= False)   # train, test separate

img= X_train[0]
label= y_train[0]
print(label)

print(img.shape)
img= img.reshape(28, 28)
print(img.shape)

img_show(img)
