import numpy as np
import cv2 as cv

cap = cv.VideoCapture("Tokyo_2020_Highlight_1.mp4")
advert = cv.imread("euro.png", -1)
advert = cv.resize(advert, (200, 100))

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Hardcoded initial points
p0 = np.array([[[677, 460]], [[1406, 512]], [[1119, 628]]], dtype=np.float32)
mask = np.zeros_like(old_frame)
black_display = np.copy(old_frame) * 0


def blend_non_transparent(face_img, overlay_img):
    # https://stackoverflow.com/questions/36921496/how-to-join-png-with-alpha-transparency-in-a-frame-in-realtime/37198079#37198079

    gray_overlay = cv.cvtColor(overlay_img, cv.COLOR_BGR2GRAY)
    overlay_mask = cv.threshold(gray_overlay, 1, 255, cv.THRESH_BINARY)[1]
    overlay_mask = cv.erode(overlay_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv.blur(overlay_mask, (3, 3))
    background_mask = 255 - overlay_mask
    overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
    background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    return np.uint8(cv.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"X:{x}, Y:{y}")


while True:
    ret, frame = cap.read()
    if ret:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.line(img, p1[0][0].astype(np.int64), p1[1][0].astype(np.int64), (255, 255, 0), 2)
        cv.line(img, p1[1][0].astype(np.int64), p1[2][0].astype(np.int64), (255, 255, 0), 2)
        cv.line(img, p1[2][0].astype(np.int64), p1[0][0].astype(np.int64), (255, 255, 0), 2)

        # advert = cv.copyMakeBorder(advert, 0, black_display.shape[0] - advert.shape[0], 0, black_display.shape[1] - advert.shape[1], 0)
        # advert_placement = place_advert(black_display, advert, p1[0][0])
        cv.imshow('frame', img)
        # cv.imshow('advert', advert)
        # cv.imshow('advert2', advert_placement)
        cv.setMouseCallback("frame", mouse_click)
        k = cv.waitKey(0)
    else:
        pass
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
