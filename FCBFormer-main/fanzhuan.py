# import cv2
# import numpy as np
#
# # 加载一维图像
# input_image_path = "img.png"
# image = cv2.imread(input_image_path)
#
# # 进行模糊处理
# blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # 使用高斯模糊作为示例
#
# # 保存模糊后的图像
# output_image_path = "blurred_image.jpg"
# cv2.imwrite(output_image_path, blurred_image)
#
# # 显示模糊后的图像（可选）
# cv2.imshow("Blurred Image", blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 加载模糊图像
blurred_image_path = "blurred_image.jpg"
blurred_image = cv2.imread(blurred_image_path, cv2.IMREAD_GRAYSCALE)

# 对模糊图像应用sigmoid函数
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x / 255.0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

blurred_image_sigmoid = sigmoid((blurred_image - np.mean(blurred_image)) / np.std(blurred_image))


# blurred_image_sigmoid = sigmoid(blurred_image * 10)
cv2.imshow("Segmentation Result", blurred_image_sigmoid)
cv2.waitKey(0)
# 将图像进行二值化，以创建分割区域
_, segmented_image = cv2.threshold(blurred_image_sigmoid, 0.5, 255, cv2.THRESH_BINARY)



segmented_image = np.uint8(segmented_image)
segmented_image = cv2.bitwise_not(segmented_image)

# output_image_path = "fanzhuan3.jpg"
# cv2.imwrite(output_image_path, segmented_image)
# 显示结果图像
cv2.imshow("Segmentation Result", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


