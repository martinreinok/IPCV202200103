import cv2
import numpy as np
import cv2 as cv

cap = cv.VideoCapture("Tokyo_2020_Highlight_1.mp4")
advert = cv.imread("euro.png", -1)
advert = cv.resize(advert, (1920, 1080))
videowriter = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (1920, 1080))
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Hardcoded initial points
# p0 = np.array([[[677, 460]], [[1406, 512]], [[1119, 628]], [[1076, 625]]], dtype=np.float32)
p0 = np.array([[[963, 299]], [[1061, 305]], [[1058, 359]], [[963, 355]], [[677, 460]]], dtype=np.float32)
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
    @staticmethod
    def pixel2world(x, y, homography):
        # https://stackoverflow.com/questions/44578876/opencv-homography-to-find-global-xy-coordinates-from-pixel-xy-coordinates
        imagepoint = [x, y, 1]
        worldpoint = np.array(np.dot(homography, imagepoint))
        scalar = worldpoint[2]
        xworld = worldpoint[0] / scalar
        yworld = worldpoint[1] / scalar
        return xworld, yworld, scalar

    @staticmethod
    def world2pixel(coords, homography):
        # https://stackoverflow.com/questions/44578876/opencv-homography-to-find-global-xy-coordinates-from-pixel-xy-coordinates
        imagepoint = coords
        worldpoint = np.array(np.dot(np.linalg.inv(homography), imagepoint))
        scalar = worldpoint[2]
        x_pixel = worldpoint[0] / scalar
        y_pixel = worldpoint[1] / scalar
        return int(x_pixel), int(y_pixel), scalar


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
            court_corner_left = p1[4][0].astype(np.int64)
            points_coords_3d = np.asarray([[0, 122, 0], [182, 122, 0], [182, 0, 0], [0, 0, 0], [-660, -270, 0]])
            cv.line(img, backboard_top_left, backboard_top_right, (255, 255, 0), 2)
            cv.line(img, backboard_top_right, backboard_bot_right, (255, 255, 0), 2)
            cv.line(img, backboard_bot_right, backboard_bot_left, (255, 255, 0), 2)
            cv.line(img, backboard_bot_left, backboard_top_left, (255, 255, 0), 2)

            homography = cv.findHomography(p1, points_coords_3d, 0, 0)[0]
            world_coords = Projection.pixel2world(court_corner_left[0], court_corner_left[1], homography)
            print(f"Court corner world: {world_coords}")

            advert_world = [[-550, -260, 1], [-550, -160, 1.1], [-380, -160, 1.1], [-380, -260, 0.97]]
            advert_bot_left = Projection.world2pixel(advert_world[0], homography)
            advert_top_left = Projection.world2pixel(advert_world[1], homography)
            advert_top_right = Projection.world2pixel(advert_world[2], homography)
            advert_bot_right = Projection.world2pixel(advert_world[3], homography)
            # Draw temporary rectangle
            cv.line(img, advert_bot_left[:2], advert_top_left[:2], (255, 100, 0), 2)
            cv.line(img, advert_top_left[:2], advert_top_right[:2], (255, 100, 0), 2)
            cv.line(img, advert_top_right[:2], advert_bot_right[:2], (255, 100, 0), 2)
            cv.line(img, advert_bot_right[:2], advert_bot_left[:2], (255, 100, 0), 2)

            cv.line(img, advert_bot_right[:2], advert_top_left[:2], (255, 100, 0), 1)
            cv.line(img, advert_top_right[:2], advert_bot_left[:2], (255, 100, 0), 1)

            cv.imshow('frame', img)
            videowriter.write(img)
            cv.setMouseCallback("frame", mouse_click)
            k = cv.waitKey(10)
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            cv.destroyAllWindows()
            videowriter.release()
            break
    except Exception as error:
        print(error)
        cv.destroyAllWindows()
