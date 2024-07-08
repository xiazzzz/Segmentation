import cv2
import matplotlib.pyplot as plt

# 读取图片
image_path = '1.jpg'  # 将路径替换为你的图像路径
image = cv2.imread(image_path)

# 将图片从 BGR 格式转换为 RGB 格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 创建三个子图，分别显示红色通道、绿色通道和蓝色通道
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示红色通道
axes[0].imshow(image_rgb[:, :, 0], cmap='gray')
axes[0].axis('off')
# 显示绿色通道
axes[1].imshow(image_rgb[:, :, 1], cmap='gray')
axes[1].axis('off')
# 显示蓝色通道
axes[2].imshow(image_rgb[:, :, 2], cmap='gray')
axes[2].axis('off')
plt.show()