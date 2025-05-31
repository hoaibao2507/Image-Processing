import cv2
import numpy as np

L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    f = np.zeros((P, Q), np.float64)
    f[0:M, 0:N] = imgin.astype(np.float64) / (L - 1)
    F = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    F = np.fft.fftshift(F)
    S = cv2.magnitude(F[:, :, 0], F[:, :, 1])
    S = np.clip(S, 0, L - 1).astype(np.uint8)
    return S

def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    f = np.zeros((P, Q), np.float32)
    f[0:M, 0:N] = imgin

    F = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    F = np.fft.fftshift(F)

    H = np.zeros((P, Q, 2), np.float32)
    n = 2
    D0 = 60
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u - P // 2) ** 2 + (v - Q // 2) ** 2)
            if Duv > 0:
                H[u, v, 0] = 1 / (1 + np.power(D0 / Duv, 2 * n))

    G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)
    G = np.fft.ifftshift(G)
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    g = g[0:M, 0:N, 0]

    g = np.clip(g, 0, L - 1).astype(np.uint8)
    return g

def CreateNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P, Q), np.float32)
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            for (ux, vx) in [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]:
                Duv = np.sqrt((u - ux) ** 2 + (v - vx) ** 2)
                if Duv > 0:
                    h *= 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
                else:
                    h *= 0.0
                Duv = np.sqrt((u - (P - ux)) ** 2 + (v - (Q - vx)) ** 2)
                if Duv > 0:
                    h *= 1.0 / (1.0 + np.power(D0 / Duv, 2 * n))
                else:
                    h *= 0.0
            H[u, v] = h
    return H

def DrawNotchRejectFilter(_=None):
    H = CreateNotchRejectFilter()
    H = H * (L - 1)
    H = H.astype(np.uint8)
    return H

def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    H = CreateNotchRejectFilter()
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u, v, 0] = F[u, v, 0] * H[u, v]
            G[u, v, 1] = F[u, v, 1] * H[u, v]

    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    gp = g[:, :, 0]

    for x in range(0, P):
        for y in range(0, Q):
            if (x + y) % 2 == 1:
                gp[x, y] = -gp[x, y]

    imgout = gp[0:M, 0:N]
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout
