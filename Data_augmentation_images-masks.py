import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from PIL import Image

def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)

image = cv2.imread(r'C:\Users\aless\Desktop\46_manual1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image0 = cv2.imread(r'C:\Users\aless\Desktop\46_training.jpg')
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)


transform1 = A.Compose([A.HorizontalFlip(p=0.5)], additional_targets={'image0': 'image'})
transform2 = A.Compose([A.ShiftScaleRotate(p=0.5)], additional_targets={'image0': 'image'})
transform3 = A.Compose([A.RandomBrightnessContrast(p=0.2)], additional_targets={'image0': 'image'})
transform4 = A.Compose([A.RGBShift(p=0.2)], additional_targets={'image0': 'image'})
transform5 = A.Compose([A.Blur(p=0.5)], additional_targets={'image0': 'image'})

random.seed(42)

transformed1 = transform1(image=image, image0=image0)
transformed2 = transform2(image=image, image0=image0)
transformed3 = transform3(image=image, image0=image0)
transformed4 = transform4(image=image, image0=image0)
transformed5 = transform5(image=image, image0=image0)

#visualize(transformed1['image0'])
#visualize(transformed1['image'])
#visualize(transformed2['image0'])
#visualize(transformed2['image'])
#visualize(transformed3['image0'])
#visualize(transformed3['image'])
#visualize(transformed4['image0'])
#visualize(transformed4['image'])
#visualize(transformed5['image0'])
#visualize(transformed5['image'])

plt.show()


cv2.imwrite('78_test.jpg', transformed1['image0'])
cv2.imwrite('78_manual1.png', transformed1['image'][:,:,0])
cv2.imwrite('79_test.jpg', transformed2['image0'])
cv2.imwrite('79_manual1.png', transformed2['image'][:,:,0])
#cv2.imwrite('53_training.jpg', transformed3['image0'])
#cv2.imwrite('53_manual1.png', transformed3['image'][:,:,0])
cv2.imwrite('80_test.jpg', transformed4['image0'])
cv2.imwrite('80_manual1.png', transformed4['image'][:,:,0])
#cv2.imwrite('55_training.jpg', transformed4['image0'])
#cv2.imwrite('55_manual1.png', transformed4['image'][:,:,0])

#attributi di A.Compose https://pypi.org/project/albumentations/