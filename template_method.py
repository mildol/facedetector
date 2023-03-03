import cv2
import matplotlib.pyplot as plt
# загрузка изображения
img_large = cv2.imread('mouth.jpg')

# загрузка шаблона
img_small = cv2.imread('mil_1.jpg')

# преобразование изображений к оттенкам серого
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
plt.title('large image')

plt.subplot(122)
plt.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
plt.title('template image')

print(img_large.shape)
print(img_small.shape)
(533, 533, 3)
(100, 100, 3)

# применение template matching
large_copy = img_large.copy()
res = cv2.matchTemplate(image=large_copy,
templ=img_small,
method=cv2.TM_CCOEFF)
plt.subplot(121)
plt.imshow(res, cmap='hot')

plt.subplot(122)
plt.imshow(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# построение рамки для обнаруженных лиц
w, h, c = img_small.shape

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img=large_copy,
pt1=top_left,
pt2=bottom_right,
color=(0,0,255),
thickness=5);

plt.imshow(cv2.cvtColor(large_copy, cv2.COLOR_BGR2RGB))

cv2.imwrite('matching_image.png', large_copy)
cv2.waitKey(0)
plt.show()
