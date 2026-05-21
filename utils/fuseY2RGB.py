import os
import cv2
import numpy as np

# 定义文件夹路径
def fuseY2RGB(folder1,folder2,folderf,output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filepath1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.tif'))])
    filepath2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.jpg', '.png', '.tif'))])
    filepathf = sorted([f for f in os.listdir(folderf) if f.endswith(('.jpg', '.png', '.tif'))])

    assert len(filepath1) == len(filepath2) == len(filepathf), "File counts do not match!"

    for pic, (file1, file2, filef) in enumerate(zip(filepath1, filepath2, filepathf), start=1):
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))
        imgf_y = cv2.imread(os.path.join(folderf, filef), cv2.IMREAD_GRAYSCALE).astype(np.float64)

        if img1.shape[2] == 3:
            ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb).astype(np.float64)
            cb1, cr1 = ycbcr1[:, :, 1], ycbcr1[:, :, 2]
        else:
            cb1, cr1 = None, None

        if img2.shape[2] == 3:
            ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb).astype(np.float64)
            cb2, cr2 = ycbcr2[:, :, 1], ycbcr2[:, :, 2]
        else:
            cb2, cr2 = None, None

        # 融合 Cb 和 Cr 通道
        if cb1 is not None and cb2 is not None:
            cbf = np.where((cb1 == 128) & (cb2 == 128), 128,
                           (cb1 * np.abs(cb1 - 128) + cb2 * np.abs(cb2 - 128)) / (np.abs(cb1 - 128) + np.abs(cb2 - 128)))
        elif cb1 is not None:
            cbf = cb1
        elif cb2 is not None:
            cbf = cb2
        else:
            cbf = None

        if cr1 is not None and cr2 is not None:
            crf = np.where((cr1 == 128) & (cr2 == 128), 128,
                           (cr1 * np.abs(cr1 - 128) + cr2 * np.abs(cr2 - 128)) / (np.abs(cr1 - 128) + np.abs(cr2 - 128)))
        elif cr1 is not None:
            crf = cr1
        elif cr2 is not None:
            crf = cr2
        else:
            crf = None

        print(f"[{filef}] cb1 is None: {cb1 is None}, cb2 is None: {cb2 is None}")
        # 构建融合后的 YCbCr 图像并转换回 RGB
        if cbf is not None and crf is not None:
            imgf_ycbcr = np.zeros((imgf_y.shape[0], imgf_y.shape[1], 3), dtype=np.float64)
            imgf_ycbcr[:, :, 0] = imgf_y
            imgf_ycbcr[:, :, 1] = cbf
            imgf_ycbcr[:, :, 2] = crf
            imgf_ycbcr = np.clip(imgf_ycbcr, 0, 255)
            imgf_rgb = cv2.cvtColor(imgf_ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        else:
            imgf_rgb = imgf_y.astype(np.uint8)  # 如果 Cb 和 Cr 不存在，只保存 Y 通道

        # 保存结果
        output_path = os.path.join(output_folder, filef)
        cv2.imwrite(output_path, imgf_rgb)

        # 清理变量
        del cb1, cb2, cr1, cr2, cbf, crf






import cv2
import numpy as np
import os

def fuseY2RGB_single(img1_path, img2_path, imgf_path, output_path=None):
    # 读取输入图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    imgf_y = cv2.imread(imgf_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    # 转换为 YCbCr 颜色空间
    cb1, cr1 = None, None
    if img1 is not None and img1.ndim == 3 and img1.shape[2] == 3:
        ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb).astype(np.float64)
        cb1, cr1 = ycbcr1[:, :, 1], ycbcr1[:, :, 2]

    cb2, cr2 = None, None
    if img2 is not None and img2.ndim == 3 and img2.shape[2] == 3:
        ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb).astype(np.float64)
        cb2, cr2 = ycbcr2[:, :, 1], ycbcr2[:, :, 2]

    # 融合 Cb 和 Cr 通道
    if cb1 is not None and cb2 is not None:
        cbf = np.where((cb1 == 128) & (cb2 == 128), 128,
                       (cb1 * np.abs(cb1 - 128) + cb2 * np.abs(cb2 - 128)) / (np.abs(cb1 - 128) + np.abs(cb2 - 128)))
    elif cb1 is not None:
        cbf = cb1
    elif cb2 is not None:
        cbf = cb2
    else:
        cbf = None

    if cr1 is not None and cr2 is not None:
        crf = np.where((cr1 == 128) & (cr2 == 128), 128,
                       (cr1 * np.abs(cr1 - 128) + cr2 * np.abs(cr2 - 128)) / (np.abs(cr1 - 128) + np.abs(cr2 - 128)))
    elif cr1 is not None:
        crf = cr1
    elif cr2 is not None:
        crf = cr2
    else:
        crf = None

    # 构建融合后的 YCbCr 图像并转换回 RGB
    if cbf is not None and crf is not None:
        imgf_ycbcr = np.zeros((imgf_y.shape[0], imgf_y.shape[1], 3), dtype=np.float64)
        imgf_ycbcr[:, :, 0] = imgf_y
        imgf_ycbcr[:, :, 1] = cbf
        imgf_ycbcr[:, :, 2] = crf
        imgf_ycbcr = np.clip(imgf_ycbcr, 0, 255)
        imgf_rgb = cv2.cvtColor(imgf_ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    else:
        imgf_rgb = imgf_y.astype(np.uint8)  # 只保存 Y 通道

    # 保存结果
    cv2.imwrite(imgf_path, imgf_rgb)

    return imgf_rgb
