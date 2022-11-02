import cv2
import traceback
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

video_name = "basketball_2.mp4"
folderOffset = "videos\\"
input_video = cv2.VideoCapture(folderOffset + video_name)
advertisement = cv2.imread("UTLogo.png", -1)
advertisement = cv2.cvtColor(advertisement, cv2.COLOR_RGBA2RGB)
advertisement = cv2.resize(advertisement, (1920, 1080))

""" Just some links that might be useful
https://stackoverflow.com/questions/36921496/how-to-join-png-with-alpha-transparency-in-a-frame-in-realtime/37198079#37198079
"""


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X:{x}, Y:{y}")


def calculate_point_on_line(point1, point2, distance_from_p1):
    x = int(point1[0] + abs(point1[0] - point2[0]) * distance_from_p1)
    y = int(point1[1] + abs(point1[1] - point2[1]) * distance_from_p1)
    return x, y


class Projection:
    # https://stackoverflow.com/questions/76134/how-do-i-reverse-project-2d-points-into-3d
    @staticmethod
    def pixel2world(x, y, homography_matrix):
        # https://stackoverflow.com/questions/44578876/opencv-homography-to-find-global-xy-coordinates-from-pixel-xy-coordinates
        imagepoint = [x, y, 1]
        worldpoint = np.array(np.dot(homography_matrix, imagepoint))
        scalar = worldpoint[2]
        xworld = worldpoint[0] / scalar
        yworld = worldpoint[1] / scalar
        return xworld, yworld, scalar

    @staticmethod
    def world2pixel(world_coordinates, homography_matrix):
        world_point = np.array(np.dot(np.linalg.inv(homography_matrix), world_coordinates))
        scalar = world_point[2]
        x_pixel = world_point[0] / scalar
        y_pixel = world_point[1] / scalar
        return int(x_pixel), int(y_pixel), scalar


def show_multiple_output(video_feeds_list, scale):
    for i, video_feed in enumerate(video_feeds_list):
        cv2.namedWindow(f"Output {i}", cv2.WINDOW_KEEPRATIO)
        cv2.imshow(f"Output {i}", video_feed)
        cv2.resizeWindow(f"Output {i}", int(input_video.get(3)) // scale, int(input_video.get(4)) // scale)
        cv2.setMouseCallback(f"Output {i}", mouse_click)


def hardcoded_points_selector(video_name):
    points = None
    coordinates_3d = None
    advert_world_coordinates = None
    if video_name == "basketball_1.mp4":
        # Reference to 'basketball_1_points.png'
        points = np.array([[[966, 355]], [[965, 299]], [[1063, 302]], [[1060, 358]],
                           [[680, 459]], [[1406, 510]]], dtype=np.float32)
        # X, Y, Z (METERS)
        coordinates_3d = np.array([[[0, 0, 0], [0, 1.1, 0], [1.85, 1.1, 0], [1.65, 0, 0],
                                    [1.75 / 2 - 15 / 2, -3, -1.2], [1.75 / 2 + 15 / 2, -3, -1.2]]], np.float32)
        advert_world_coordinates = np.array([[[0, 0, 0], [-3, 0, 0], [0, -3, 0], [0, 0, -3]]], np.float32)

    if video_name == "basketball_2.mp4":
        # Reference to 'basketball_1_points.png'
        points = np.array([[[853, 260]], [[849, 161]], [[1032, 166]], [[1024, 266]],
                           [[369, 432]], [[1518, 475]]], dtype=np.float32)
        # X, Y, Z (METERS)
        coordinates_3d = np.array([[[0, 0, 0], [0, 1.1, 0], [1.85, 1.1, 0], [1.65, 0, 0],
                                    [1.75 / 2 - 15 / 2, -3, -1.2], [1.75 / 2 + 15 / 2, -3, -1.2]]], np.float32)
        advert_world_coordinates = np.array([[[0, 0, 0], [-3, 0, 0], [0, -3, 0], [0, 0, -3]]], np.float32)

    if video_name == "tennis_2.mp4":
        # Reference to 'tennis_1_points.png'
        points = np.array([[[1640, 823]], [[1469, 817]], [[628, 271]], [[708, 272]],
                           [[1484, 454]], [[441, 452]]], dtype=np.float32)
        # X, Y, Z (METERS)
        coordinates_3d = np.array([[[0, 0, 0], [1.372, 0, 0], [10.973, 23.770, 0], [9.601, 23.770, 0],
                                    [-0.9, 11.885, 0], [10.973 + 1, 11.885, 0]]], np.float32)
        advert_world_coordinates = np.array([[[0, 0, 0], [11, 0, 0], [0, 24, 0], [0, 0, -2]]], np.float32)

    return points, coordinates_3d, advert_world_coordinates


Projection = Projection()
SAVE_VIDEO = 0
SELECT_POINTS_ONLY = 0
videowriter = None


def nothing(x):
    pass


if SAVE_VIDEO:
    videowriter = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (1920, 1080))

if __name__ == "__main__":
    while True:

        if SELECT_POINTS_ONLY:

            while True:
                ret, frame = input_video.read()
                show_multiple_output([frame], 1)
                k = cv2.waitKey(0)
                if k == 27:  # ESC exits the video
                    cv2.destroyAllWindows()
                    input_video.release()
                    raise SystemExit
        frame_count = 0
        input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Initialize optical flow parameters
        good_new, good_old = None, None
        lucas_kanade = dict(winSize=(15, 15), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0, 255, (100, 3))
        # Grab first frame for optical flow
        if input_video.isOpened():
            ret, old_frame = input_video.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(old_frame)
            # Hardcoded initial points (double-click on frame to print coordinates)
            p0, points_coordinates_3d, advert_world = hardcoded_points_selector(video_name)

            # Check for Z axis values, initial guess cannot have Z axis values
            initial_image_points = []
            image_points = []
            initial_coordinates_3d = []
            for count, point in enumerate(p0):
                if points_coordinates_3d[0][count][2] == 0:
                    initial_image_points.append(point[0])
                    initial_coordinates_3d.append(points_coordinates_3d[0][count])
                image_points.append(point[0])

            initial_image_points = np.vstack([[initial_image_points]])
            image_points = np.vstack([[image_points]])
            initial_coordinates_3d = np.vstack([[initial_coordinates_3d]])

            _, initialCameraMatrix, _, _, _ = cv2.calibrateCamera(initial_coordinates_3d,
                                                                  initial_image_points,
                                                                  old_gray.shape[::-1],
                                                                  None, None,
                                                                  flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

            _, CameraMatrix, dist, rotation_vec, translation_vec = cv2.calibrateCamera(points_coordinates_3d,
                                                                                       image_points,
                                                                                       old_gray.shape[::-1],
                                                                                       initialCameraMatrix, None,
                                                                                       flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_USE_INTRINSIC_GUESS)
            print(CameraMatrix)
            rotation_vec = rotation_vec[0]
            translation_vec = translation_vec[0]
        else:
            print("Could not open video")
            raise SystemExit

        """ Main video loop """
        while frame_count < input_video.get(cv2.CAP_PROP_FRAME_COUNT):
            ret, frame = input_video.read()
            try:
                if ret:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lucas_kanade)
                    # Select good points
                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                    main_frame = cv2.add(frame, mask)

                    backboard_top_left = p1[0][0].astype(np.int64)
                    backboard_top_right = p1[1][0].astype(np.int64)
                    backboard_bot_right = p1[2][0].astype(np.int64)
                    backboard_bot_left = p1[3][0].astype(np.int64)
                    # court_corner_left = p1[4][0].astype(np.int64)

                    # Draw a line around backboard tracking points
                    # cv2.line(main_frame, backboard_top_left, backboard_top_right, (255, 255, 0), 2)
                    # cv2.line(main_frame, backboard_top_right, backboard_bot_right, (255, 255, 0), 2)
                    # cv2.line(main_frame, backboard_bot_right, backboard_bot_left, (255, 255, 0), 2)
                    # cv2.line(main_frame, backboard_bot_left, backboard_top_left, (255, 255, 0), 2)

                    # New image points
                    points = []
                    for point in p0:
                        points.append(point[0])
                    image_points = np.vstack([[points]])

                    _, rotation_vec, translation_vec = cv2.solvePnP(points_coordinates_3d, image_points, CameraMatrix,
                                                                    None)

                    imgpts, jac = cv2.projectPoints(advert_world, rotation_vec, translation_vec, CameraMatrix,
                                                    None)

                    cv2.circle(main_frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), 15, (255, 255, 255), -1)
                    cv2.circle(main_frame, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), 15, (255, 0, 0), -1)
                    cv2.circle(main_frame, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), 15, (0, 255, 0), -1)
                    cv2.circle(main_frame, (int(imgpts[3][0][0]), int(imgpts[3][0][1])), 15, (0, 0, 255), -1)

                    cv2.line(main_frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])),
                             (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (255, 100, 0), 2)
                    cv2.line(main_frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])),
                             (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255, 100, 0), 2)
                    cv2.line(main_frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])),
                             (int(imgpts[3][0][0]), int(imgpts[3][0][1])), (255, 100, 0), 2)
                    # cv2.line(main_frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255, 100, 0), 2)

                    # cv2.line(main_frame, advert_bot_right[:2], advert_top_left[:2], (255, 100, 0), 1)
                    # cv2.line(main_frame, advert_top_right[:2], (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255, 100, 0), 1)
                    """
                    ret, camMTX, dist, rvecs, tvecs = cv2.calibrateCamera(points_coordinates_3d, p0,
                                                                          frame_gray.shape[::-1],
                                                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                          distCoeffs=None, cameraMatrix=firstCamMatrix)
                    
                    """
                    # Points coordinates defined in function: hardcoded_points_selector
                    # homography = cv2.findHomography(p1, points_coordinates_3d, 0, 0)[0]
                    # world_coords = Projection.pixel2world(court_corner_left[0], court_corner_left[1], homography)

                    '''
                    # Advert world coordinates defined in function: hardcoded_points_selector
                    advert_bot_left = Projection.world2pixel(advert_world[0], homography)
                    advert_top_left = Projection.world2pixel(advert_world[1], homography)
                    advert_top_right = Projection.world2pixel(advert_world[2], homography)
                    advert_bot_right = Projection.world2pixel(advert_world[3], homography)

                    # Draw temporary rectangle
                    cv2.line(main_frame, advert_bot_left[:2], advert_top_left[:2], (255, 100, 0), 2)
                    cv2.line(main_frame, advert_top_left[:2], advert_top_right[:2], (255, 100, 0), 2)
                    cv2.line(main_frame, advert_top_right[:2], advert_bot_right[:2], (255, 100, 0), 2)
                    cv2.line(main_frame, advert_bot_right[:2], advert_bot_left[:2], (255, 100, 0), 2)
                    cv2.line(main_frame, advert_bot_right[:2], advert_top_left[:2], (255, 100, 0), 1)
                    cv2.line(main_frame, advert_top_right[:2], advert_bot_left[:2], (255, 100, 0), 1)
                    '''
                    # Warp advertisement
                    """
                    aH, aW, c = advertisement.shape
                    advertPointMatrix = np.float32([[0, 0], [aW, 0], [0, aH], [aW, aH]])

                    advertLocationMatrix = np.float32(
                        [advert_top_left[:2], advert_top_right[:2], advert_bot_left[:2], advert_bot_right[:2]])
                    perspectiveMatrix = cv2.getPerspectiveTransform(advertPointMatrix, advertLocationMatrix)
                    advertWarpResult = cv2.warpPerspective(advertisement, perspectiveMatrix, (1920, 1080))

                    grayCol = cv2.cvtColor(advertWarpResult, cv2.COLOR_BGR2GRAY)  # grijswaarde plaatje voor een MASK
                    advertMask = cv2.inRange(grayCol, 1, 255)
                    # main_frame = cv2.add(main_frame, mask)
                    mainFrame = cv2.add(main_frame, advertWarpResult)
                    """
                    k = cv2.waitKey(1)
                    if k == 27:  # ESC exits the video
                        cv2.destroyAllWindows()
                        input_video.release()
                        raise SystemExit
                    if k == 32:  # Space bar pauses the video
                        cv2.waitKey(0)
                    # Show the output(s)
                    show_multiple_output([main_frame], 2)
                    if SAVE_VIDEO:
                        videowriter.write(main_frame)

                    # Copy current frame to old frame for optical flow
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
            except Exception as error:
                traceback.print_exc()
                print(error)
            frame_count += 1

        k = cv2.waitKey(0)  # Disable video autoplay each loop
        if k == 27:  # ESC
            if SAVE_VIDEO:
                videowriter.release()
            cv2.destroyAllWindows()
            input_video.release()
            raise SystemExit
