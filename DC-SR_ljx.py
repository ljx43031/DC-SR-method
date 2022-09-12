# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:55:51 2022
Differential correction based shadow removal method
@author: Administrator
"""

import os
from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#比值法
def RatioComp(f1, f2, mask, T):
    f1 = np.float64(f1)
    f2 = np.float64(f2)
    b1 = f1[:,:,0]
    g1 = f1[:,:,1]
    r1 = f1[:,:,2]
    b2 = f2[:,:,0]
    g2 = f2[:,:,1]
    r2 = f2[:,:,2]
    s1 = b1+g1+r1
    s2 = b2+g2+r2
    detaB = np.abs(b1/s1 - b2/s2) * mask
    detaG = np.abs(g1/s1 - g2/s2) * mask
    detaR = np.abs(r1/s1 - r2/s2) * mask
    
    ThresHB = np.where(detaB>T, 1, 0)
    ThresHG = np.where(detaG>T, 1, 0)
    ThresHR = np.where(detaR>T, 1, 0)
    # return ThresHB, ThresHG, ThresHR
    return ThresHB * ThresHG * ThresHR * 255

#灰度法
def GrayComp(fg, bg, deta_0, e0):
    #fg 前景的灰度图
    #bg 背景的灰度图
    #deta_0, e0
    ye = fg + deta_0
    yd = np.abs(ye - bg)
    eg = e0*(1 + yd / (bg + 1))
    r = np.where(yd>eg, 255, 0)
    return r


#bgr二重差分比较---ISTD数据库训练并统计阴影区域的均值
def ColorComp_istd(f1, f2, mask):
    # cv2.imwrite('part1.jpg', f1)
    # b - channel 0
    # g - channel 1
    # r - channel 2
    f1 = np.float64(f1)
    f2 = np.float64(f2)
    df = np.abs(f1 - f2) * mask /255
    df_b = df[:,:,0]
    df_g = df[:,:,1]
    df_r = df[:,:,2]
    
    df_b = df_b.flatten()
    df_g = df_g.flatten()
    df_r = df_r.flatten()
    
    df_bm = sum(df_b)/cv2.countNonZero(df_b)
    df_gm = sum(df_g)/cv2.countNonZero(df_g)
    df_rm = sum(df_r)/cv2.countNonZero(df_r)
    
    return df_bm, df_gm, df_rm
#bgr二重差分比较
def ColorComp(f1, f2, mask, b2r_c, b2r_b, b2g_c, b2g_b, g2r_c, g2r_b):
    # cv2.imwrite('part1.jpg', f1)
    # b - channel 0
    # g - channel 1
    # r - channel 2
    f1 = np.float64(f1)
    f2 = np.float64(f2)
    df = np.abs(f1 - f2)
    df_b = df[:,:,0]
    df_g = df[:,:,1]
    df_r = df[:,:,2]
    
    df_rb = np.abs(df_r - (b2r_b + b2r_c*df_b)) * mask
    df_gb = np.abs(df_g - (b2g_b + b2g_c*df_b)) * mask
    df_rg = np.abs(df_r - (g2r_b + g2r_c*df_g)) * mask
    return df_rb, df_gb, df_rg

def DC_ShadowRemoval(foreground, background, mask, Trb, Tgb, Trg):
    #foreground, background分别是含目标的前景图和不含目标的背景图，3通道彩色
    
    #bgr二重差分比较----参数
    b2r_c = 1.26    #系数
    b2r_b = 0.39    #偏差
    b2g_c = 1.10
    b2g_b = 0.85
    g2r_c = 1.11
    g2r_b = 0.11
    
    df_rb, df_gb, df_rg = ColorComp(foreground, background, mask, b2r_c, b2r_b, b2g_c, b2g_b,  g2r_c, g2r_b)
    #二重差分结果的联合：
    
    # # 三通道对目标像素判断的累加
    # ddf_rb = np.where(df_rb>Trb, 1, 0) * 85
    # ddf_gb = np.where(df_gb>Tgb, 1, 0) * 85
    # ddf_rg = np.where(df_rg>Trg, 1, 0) * 85
    # sr_r = ddf_rb + ddf_gb + ddf_rg
    
    # # 三通道对目标像素分别判断
    ddf_rb = np.where(df_rb>Trb, 1, 0)
    ddf_gb = np.where(df_gb>Tgb, 1, 0)
    ddf_rg = np.where(df_rg>Trg, 1, 0)
    # 三通道都判断为目标像素才认为是目标像素
    sr_r = ddf_rb * ddf_gb * ddf_rg *255
    # # 三通道只要有一个通道判断为目标像素就认为是目标像素   
    # sr_r = (ddf_rb | ddf_gb | ddf_rg) *255
    
    return sr_r

    # return df_rb, df_gb, df_rg
    
def mouse_callback(event, x, y, flags, param):
# cv.EVENT_LBUTTONDOWN表示鼠标左键向下点击一下
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append(np.array([x, y],'int32'))
        cv2.circle(param[0], (x, y), 3, (0, 0, 255), -1)
        # cv2.putText(param[0], str(x) + "," + str(y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("foreground", param[0])


if __name__ == "__main__":
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

    print('----------------------------------')
    print('算法检测开始')
    print('----------------------------------')
    path = 'D:/program/DC-SR/Indoors2/'
    
    #前景
    foreground = cv2.imread(path+'foreground.png')
    #背景
    background = cv2.imread(path+'background.png')
    # #前景
    # foreground = cv2.imread(path+'fg.jpg')
    # #背景
    # background = cv2.imread(path+'bg.jpg')
    foreground = cv2.resize(foreground,(640,480),interpolation=cv2.INTER_CUBIC)
    background = cv2.resize(background,(640,480),interpolation=cv2.INTER_CUBIC)
    #灰度化并且滤波去噪
    gray_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    gray_fg = cv2.GaussianBlur(gray_fg, (21, 21), 0)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)
    
    
    #提取目标区域掩码
    diff = cv2.absdiff(gray_fg, gray_bg)
    diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)
    #目标出现后所有变化区域的掩码
    mask = diff / 255
    #这是我们的差值法检测结果
    r1_s = time.time()
    r1 = DC_ShadowRemoval(foreground, background, mask, 8, 8, 8)
    r1_e = time.time()
    r1_time = r1_e - r1_s
    r1 = np.uint8(r1)
    cv2.imshow("r1", r1)
    #这是对比的比值法检测结果
    r2_s = time.time()
    r2= RatioComp(foreground, background, mask, 0.008)
    r2_e = time.time()
    r2_time = r2_e - r2_s
    r2 = np.uint8(r2)
    cv2.imshow("r2", r2)
    #灰度法检测结果
    r3_s = time.time()
    r3= GrayComp(gray_fg, gray_bg, 0.1, 35)
    r3_e = time.time()
    r3_time = r3_e - r3_s
    r3 = np.uint8(r3)
    cv2.imshow("r3", r3)
    
    # #--------------------------------------------------------------------------
    # #像素值分布情况检测
    # d1 = t1.flatten()
    # hist1, bin_edges1 = np.histogram(d1, bins=100, range=(0.01,d1.max()))
    # d2 = t2.flatten()
    # hist2, bin_edges2 = np.histogram(d2, bins=100, range=(0.01,d2.max()))
    # plt.figure()
    # plt.plot(bin_edges1[1:], hist1, color = "red")
    # plt.plot(bin_edges2[1:], hist2, color = "black")
    # #--------------------------------------------------------------------------
    
    #通过描点确定目标轮廓
    param = [foreground.copy()]
    cv2.imshow("foreground", param[0])
    cv2.setMouseCallback("foreground", mouse_callback, param=param)
    cv2.waitKey(0)
    # print(param[1:])
    target_contours = [np.expand_dims(np.asarray(param[1:]), axis=1),]
    #目标区域掩码
    cv2.drawContours(param[0], target_contours, 0, (0, 0, 255), 2)
    target_mask = np.zeros_like(foreground)
    cv2.imshow("foreground", param[0])    
    target_mask = cv2.drawContours(target_mask, target_contours, 0, 1, -1)
    target_mask = target_mask[:,:,0] * mask
    cv2.imshow("target mask", target_mask)
    shadow_mask = mask - target_mask
    cv2.imshow("shadow mask", shadow_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #定量评判：
    # 1----------目标区域中对目标像素点检测的正确率：
    target_points = np.sum(target_mask)
    #我们的方法结果：
    target_cr_r1 = np.sum((r1 * target_mask)/255) / target_points
    target_cr_r2 = np.sum((r2 * target_mask)/255) / target_points
    target_cr_r3 = np.sum((r3 * target_mask)/255) / target_points
    print("目标区域中对目标像素点检测的正确率: r1: %g; r2: %g; r3: %g" % (target_cr_r1,target_cr_r2,target_cr_r3))
    # 2----------在阴影区域中对阴影像素点检测的正确率。
    shadow_points = np.sum(shadow_mask)
    #我们的方法结果：
    shadow_cr_r1 = 1 - np.sum((r1 * shadow_mask)/255) / shadow_points
    shadow_cr_r2 = 1 - np.sum((r2 * shadow_mask)/255) / shadow_points
    shadow_cr_r3 = 1 - np.sum((r3 * shadow_mask)/255) / shadow_points
    print("阴影区域中对阴影像素点检测的正确率: r1: %g; r2: %g; r3: %g" % (shadow_cr_r1,shadow_cr_r2,shadow_cr_r3))
    
    value = input("是否需要保存图像？Y or N:")
    if value == "Y":
        cv2.imwrite(path+"r1.png", r1)
        cv2.imwrite(path+"r2.png", r2)
        cv2.imwrite(path+"r3.png", r3)
        cv2.imwrite(path+"background.png", background)
        cv2.imwrite(path+"foreground.png", foreground)
        cv2.imwrite(path+"foreground_with_target_mask.png", param[0])