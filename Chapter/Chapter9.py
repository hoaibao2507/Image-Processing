import cv2
import numpy as np

L = 256

def ConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L - 1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += color

    return label.astype(np.uint8), dem - 1  # Trả về số lượng thành phần liên thông (bỏ nền)

def CountRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L - 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += color

    max_val = max(a[1:])
    xoa = np.array([r for r in range(1, dem) if a[r] < 0.5 * max_val], dtype=np.int32)

    for x in range(M):
        for y in range(N):
            r = label[x, y]
            if r > 0:
                r -= color
                if r in xoa:
                    label[x, y] = 0

    rice_count = dem - len(xoa) - 1  # trừ 1 để bỏ background
    return label.astype(np.uint8), rice_count
