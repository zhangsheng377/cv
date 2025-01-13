import cv2
import numpy as np
import matplotlib.pyplot as plt

# image1_path = "data/33842604_1b.png"
# image2_path = "data/35593299_3a.png"
image1_path = "data/30867423_2d.png"
image2_path = "data/30867423_6j.png"

# 读取图像
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 转换为灰度图像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 使用 SIFT 提取特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 使用 FLANN 匹配器进行特征匹配
index_params = dict(algorithm=1, trees=10)  # FLANN 匹配器的参数
search_params = dict(checks=50)  # 搜索的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 使用 Lowe 的比率测试来过滤匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 计算相似度
similarity = len(good_matches) / len(keypoints1) if len(keypoints1) > 0 else 0
print(f"匹配到的良好特征点比例: {similarity}")

# 绘制匹配的特征点
result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Feature Matching with SIFT')
plt.show()
