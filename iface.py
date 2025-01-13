import os
from datetime import datetime

import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ransac import ransac as compare_images

# 指定保存图像的目录
save_dir = "saved_images"

# 如果目录不存在，创建目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def process_images(image1, image2):
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用时间戳创建唯一的文件名
    image1_filename = os.path.join(save_dir, f"image1_{timestamp}.jpg")
    image2_filename = os.path.join(save_dir, f"image2_{timestamp}.jpg")
    cv2.imwrite(image1_filename, image1)  # 保存第一张图
    cv2.imwrite(image2_filename, image2)  # 保存第二张图

    return compare_images(image1, image2)


# Gradio 接口
iface = gr.Interface(
    fn=process_images,
    inputs=[gr.Image(type="numpy", label="上传第一张图片"),
            gr.Image(type="numpy", label="上传第二张图片")],
    outputs=[gr.Textbox(label="匹配结果"),
             gr.Image(type="numpy", label="匹配特征点的图片")],
    live=True,
    title="图像特征点匹配",
    description="上传两张图片，程序将计算它们的相似度，并标记出匹配的特征点。",
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
