import cv2
import numpy as np

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 加载一维图像
input_image_path = "blurred_image.jpg"
image = cv2.imread(input_image_path)

# # 进行更强烈的模糊处理，增加高斯核的大小
# blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# 将图像转换为灰度

# 对灰度图像应用Sigmoid函数
contrast_image = sigmoid((image - np.mean(image)) / np.std(image))

# 将Sigmoid后的图像映射回0-255范围
# contrast_image = (contrast_image * 255).astype(np.uint8)

# 显示Sigmoid后的图像（可选）
cv2.imshow("Contrast Image", contrast_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
