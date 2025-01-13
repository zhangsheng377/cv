import cv2
import numpy as np
import matplotlib.pyplot as plt

image1_path = "data/33842604_1b.png"
image2_path = "data/35593299_3a.png"


# image1_path = "data/30867423_2d.png"
# image2_path = "data/30867423_6j.png"


def ransac(image1, image2):
    # 转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT 提取特征
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 使用 FLANN 匹配器进行特征匹配
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches1 = flann.knnMatch(descriptors1, descriptors2, k=2)
    matches2 = flann.knnMatch(descriptors2, descriptors1, k=2)

    # 使用 Lowe 的比率测试来过滤匹配
    good_matches1 = []
    good_matches2 = []

    for m, n in matches1:
        if m.distance < 0.7 * n.distance:
            good_matches1.append(m)

    for m, n in matches2:
        if m.distance < 0.7 * n.distance:
            good_matches2.append(m)

    # 计算双向匹配的得分
    score1 = len(good_matches1)  # A -> B 的匹配得分
    score2 = len(good_matches2)  # B -> A 的匹配得分

    # 选择匹配得分较高的方向
    if score1 >= score2:
        good_matches = good_matches1
        keypoints_1 = keypoints1
        keypoints_2 = keypoints2
        image_1 = image1
        image_2 = image2
    else:
        good_matches = good_matches2
        keypoints_1 = keypoints2
        keypoints_2 = keypoints1
        image_1 = image2
        image_2 = image1

    # 获取良好匹配的特征点坐标
    points1 = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches])

    try:
        # 使用 RANSAC 估计单应性矩阵（Homography Matrix）
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        # 通过 mask 筛选出符合单应性变换的点
        good_points = []
        for i, m in enumerate(good_matches):
            if mask[i]:  # 只保留符合 RANSAC 的点
                good_points.append(m)
    except:
        good_points = []

    # 计算良好匹配点是否符合同一映射关系
    matching_ratio = len(good_points) / len(good_matches) if len(good_matches) > 0 else 0
    result = f"匹配的比例: {matching_ratio}"
    # 绘制匹配的特征点
    result_image = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, good_points, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result, result_image


if __name__ == "__main__":
    # 读取图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    result, result_image = ransac(image1, image2)
    print(result)

    # 显示匹配结果
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching with Homography')
    plt.show()
