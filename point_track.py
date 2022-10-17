import cv2
import numpy as np
import cv2 as cv

cap = cv.VideoCapture("Tokyo_2020_Highlight_1.mp4")
advert = cv.imread("euro.png", -1)
advert = cv.resize(advert, (1920, 1080))

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Hardcoded initial points
# p0 = np.array([[[677, 460]], [[1406, 512]], [[1119, 628]], [[1076, 625]]], dtype=np.float32)
p0 = np.array([[[963, 299]], [[1061, 305]], [[1058, 359]], [[963, 355]]], dtype=np.float32)
"""Court left point, Court right point, free throw point, """
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


def calculate_point_on_line(point1, point2, distance_from_p1):
    x = int(point1[0] + abs(point1[0] - point2[0]) * distance_from_p1)
    y = int(point1[1] + abs(point1[1] - point2[1]) * distance_from_p1)
    return x, y


class Projection:
    # https://stackoverflow.com/questions/76134/how-do-i-reverse-project-2d-points-into-3d
    pass


def estimate_camera_matrix(points, image):
    c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    c34 = 1
    x_length = 100
    y_length = 40
    x1 = points[0][0][0]
    y1 = points[0][0][1]
    x2 = points[1][0][0]
    y2 = points[1][0][1]
    x3 = points[2][0][0]
    y3 = points[2][0][1]

    uv_matrix = np.array([[x1], [y1], [x2], [y2], [x3], [y3]])

    c_matrix = np.array([[c11], [c12], [c13], [c14], [c21], [c22],
                         [c23], [c24], [c31], [c32], [c33], [c34]])

    perspective_1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    perspective_2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    perspective_3 = np.array([x_length, 0, 0, 1, 0, 0, 0, 0, -x2 * x_length, 0, 0])
    perspective_4 = np.array([0, 0, 0, 0, x_length, 0, 0, 1, -y2 * x_length, 0, 0])
    perspective_5 = np.array([0, y_length, 0, 1, 0, 0, 0, 0, 0, -x3 * y_length, 0])
    perspective_6 = np.array([0, 0, 0, 0, 0, y_length, 0, 1, 0, -y3 * y_length, 0])

    perspective_matrix = np.array([perspective_1, perspective_2, perspective_3,
                                   perspective_4, perspective_5, perspective_6])

    var_c, residuals, rank, s = np.linalg.lstsq(perspective_matrix, uv_matrix, rcond=None)

    c11 = var_c[0]
    c12 = var_c[1]
    c13 = var_c[2]
    c14 = var_c[3]
    c21 = var_c[4]
    c22 = var_c[5]
    c23 = var_c[6]
    c24 = var_c[7]
    c31 = var_c[8]
    c32 = var_c[9]
    c33 = var_c[10]

    xas1 = 0
    yas1 = 0
    xas2 = 20
    yas2 = 20
    xas3 = 10
    yas3 = 10
    zas = 0

    unew1 = int((c11 * xas1 + c12 * yas1 + c13 * zas + c14) / (c31 * xas1 + c32 * yas1 + c33 * zas + 1))
    vnew1 = int((c21 * xas1 + c22 * yas1 + c23 * zas + c24) / (c31 * xas1 + c32 * yas1 + c33 * zas + 1))
    unew2 = int((c11 * xas2 + c12 * yas2 + c13 * zas + c14) / (c31 * xas2 + c32 * yas2 + c33 * zas + 1))
    vnew2 = int((c21 * xas2 + c22 * yas2 + c23 * zas + c24) / (c31 * xas2 + c32 * yas2 + c33 * zas + 1))
    unew3 = int((c11 * xas3 + c12 * yas3 + c13 * zas + c14) / (c31 * xas3 + c32 * yas3 + c33 * zas + 1))
    vnew3 = int((c21 * xas3 + c22 * yas3 + c23 * zas + c24) / (c31 * xas3 + c32 * yas3 + c33 * zas + 1))

    cv.circle(image, (unew1, vnew1), 10, (255, 0, 255), -1)
    cv.circle(image, (unew2, vnew2), 10, (255, 0, 255), -1)
    cv.circle(image, (unew3, vnew3), 10, (255, 0, 255), -1)

    # print((unew1, vnew1))
    # print((unew2, vnew2))
    # print((unew3, vnew3))


Projection = Projection()
while True:
    ret, frame = cap.read()
    try:
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

            backboard_top_left = p1[0][0].astype(np.int64)
            backboard_top_right = p1[1][0].astype(np.int64)
            backboard_bot_right = p1[2][0].astype(np.int64)
            backboard_bot_left = p1[3][0].astype(np.int64)
            backboard_size_3d = np.asarray([[0, 122, 0], [182, 122, 0], [182, 0, 0], [0, 0, 0]])

            cv.line(img, backboard_top_left, backboard_top_right, (255, 255, 0), 2)
            cv.line(img, backboard_top_right, backboard_bot_right, (255, 255, 0), 2)
            cv.line(img, backboard_bot_right, backboard_bot_left, (255, 255, 0), 2)
            cv.line(img, backboard_bot_left, backboard_top_left, (255, 255, 0), 2)

            advert_size = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])
            backboard_shape = np.float32([backboard_top_left, backboard_top_right,
                                          backboard_bot_right, backboard_bot_left])
            transform = cv.getPerspectiveTransform(advert_size, backboard_shape)
            advert2 = cv.warpPerspective(advert, transform, (1920, 1080))
            # final_image = blend_non_transparent(img, advert2)
            result = cv2.copyTo(frame, frame)
            result += cv2.copyTo(advert2, advert2)

            cv.imshow('advert', result)
            # homography = cv.findHomography(p1, backboard_size_3d, 0, 0)
            # print(homography[0])
            # try:
            #     im_out = cv2.warpPerspective(advert, homography[0], (advert.shape[1] * 2, advert.shape[0] * 2))
            #     cv.imshow('out', im_out)
            # except:
            #     pass
            # print(matrix1)
            # print(mask1)

            # advert_point_0 = calculate_point_on_line(p1[0][0].astype(np.int64), p1[1][0].astype(np.int64), 0)
            # advert_point_1 = calculate_point_on_line(p1[0][0].astype(np.int64), p1[1][0].astype(np.int64), 0.15)
            # cv.circle(img, advert_point_0, 7, (255, 100, 100), -1)
            # cv.circle(img, advert_point_1, 7, (255, 100, 100), -1)

            cv.imshow('frame', img)
            # cv.imshow('advert', advert)
            # cv.imshow('advert', advert)
            # cv.imshow('advert2', advert_placement)
            cv.setMouseCallback("frame", mouse_click)
            k = cv.waitKey(1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            cv.destroyAllWindows()
            break
    except Exception as error:
        print(error)
        cv.destroyAllWindows()
