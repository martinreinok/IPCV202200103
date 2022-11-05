import cv2
import traceback
import numpy as np
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
video_name = "basketball_2.mp4"
folderOffset = "videos\\"
input_video = cv2.VideoCapture(folderOffset + video_name)
advertisement = cv2.imread("UTLogo.png", -1)
advertisement = cv2.cvtColor(advertisement, cv2.COLOR_RGBA2RGB)
advertisement = cv2.resize(advertisement, (1920, 1080))
shadow = cv2.imread("shadow.jpg")
shadow = cv2.cvtColor(shadow, cv2.COLOR_RGBA2RGB)
shadow = cv2.resize(shadow, (1920, 1080))


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X:{x}, Y:{y}")


def show_multiple_output(video_feeds_list, scale):
    for i, video_feed in enumerate(video_feeds_list):
        cv2.namedWindow(f"Output {i}", cv2.WINDOW_KEEPRATIO)
        cv2.imshow(f"Output {i}", video_feed)
        cv2.resizeWindow(f"Output {i}", int(input_video.get(3)) // scale, int(input_video.get(4)) // scale)
        cv2.setMouseCallback(f"Output {i}", mouse_click)


def optical_flow_point_selector(video_name: str):
    points = None
    coordinates_3d = None
    advert_world_coordinates = None
    shadow_1_coords = None
    advert_2_world_coordinates = None

    if "basketball" in video_name:

        """
        Each video has predetermined starting coordinates, which are tracked using optical flow
        Basketball videos have different tracking points however same advertisement position.
        """
        if video_name == "basketball_1.mp4":
            # Reference to 'basketball_1_points.png'
            points = np.array([[[966, 355]], [[965, 299]], [[1063, 302]], [[1060, 358]],
                               [[680, 459]], [[1406, 510]]], dtype=np.float32)
        if video_name == "basketball_2.mp4":
            # Reference to 'basketball_1_points.png'
            points = np.array([[[853, 260]], [[849, 161]], [[1025, 172]], [[1022, 266]],
                               [[369, 432]], [[1518, 475]]], dtype=np.float32)

        # X, Y, Z (METERS)
        coordinates_3d = np.array([[[0, 0, 0], [0, 1.1, 0], [1.85, 1.1, 0], [1.65, 0, 0],
                                    [1.75 / 2 - 15 / 2, -2.9, -1.2], [1.75 / 2 + 15 / 2, -2.9, -1.2]]], np.float32)
        advert_world_coordinates = np.array([[[-6, -3, -1.5], [-6, -2.134, -2], [-3, -2.134, -2], [-3, -3, -1.5]]],
                                            np.float32)
        shadow_1_coords = np.array([[[-6.1, -3, -1.4], [-6, -3, -3], [-2.95, -3, -3], [-2.95, -3, -1.4]]],
                                   np.float32)
        advert_2_world_coordinates = np.array([[[4, -3, -2], [4, -3, -3], [7, -3, -3], [7, -3, -2]]],
                                              np.float32)
    if "tennis" in video_name:
        if video_name == "tennis_1.mp4":
            # Reference to 'tennis_1_points.png'
            points = np.array([[[1671, 806]], [[1508, 803]], [[716, 270]], [[788, 271]],
                               [[1498, 444]], [[539, 430]]], dtype=np.float32)

        if video_name == "tennis_2.mp4":
            # Reference to 'tennis_1_points.png'
            points = np.array([[[1640, 823]], [[1469, 817]], [[628, 271]], [[708, 272]],
                               [[1484, 454]], [[441, 452]]], dtype=np.float32)

        # X, Y, Z (METERS)
        coordinates_3d = np.array([[[0, 0, 0], [1.372, 0, 0], [10.973, 23.77, 0], [9.601, 23.77, 0],
                                    [-1, 23.77 / 2, 0], [11.973, 23.77 / 2, 0]]], np.float32)
        advert_world_coordinates = np.array(
            [[[11.6, 1, 0], [11.5 + 0.577, 1, -1], [11.5 + 0.577, 5, -1], [11.5, 5, 0]]], np.float32)
        shadow_1_coords = np.array([[[11.6, 1, 0], [13, 2.5, 0], [13, 6.5, 0], [11.5, 5, 0]]],
                                   np.float32)
        advert_2_world_coordinates = np.array([[[9, 1, 0], [9, 4, 0], [3, 4, 0], [3, 1, 0]]],
                                              np.float32)

    return points, coordinates_3d, advert_world_coordinates, advert_2_world_coordinates, shadow_1_coords


def create_hsv_mask(hsv_filter_frame, hsv_low, hsv_high):
    hsv = cv2.cvtColor(hsv_filter_frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([hsv_low[0], hsv_low[1], hsv_low[2]])
    higher_hsv = np.array([hsv_high[0], hsv_high[1], hsv_high[2]])
    hsv_mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    return hsv_mask


SAVE_VIDEO = False
SELECT_POINTS_ONLY = 0
videowriter = None
SHOW_TRACKING_POINTS = True
PRINT_EXECUTION_TIME = True
# Turn off advertisements
SHOW_ADVERTISEMENT_1 = True  # Also includes shadow
SHOW_ADVERTISEMENT_2 = True


def nothing(x):
    pass


if SAVE_VIDEO:
    videowriter = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (1920, 1080))

if __name__ == "__main__":
    while True:

        if SELECT_POINTS_ONLY:
            while True:
                ret, frame = input_video.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                color = np.random.randint(0, 255, (100, 3))
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                for i, point in enumerate(p0):
                    frame = cv2.circle(frame, (int(point[0][0]), int(point[0][1])), 15, color[i].tolist(), 3)

                show_multiple_output([frame], 1)
                k = cv2.waitKey(0)
                if k == 27:  # ESC exits the video
                    cv2.destroyAllWindows()
                    input_video.release()
                    raise SystemExit
        frame_count = 0
        input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        good_new, good_old = None, None
        """
        Lukas-Kanade parameters are defined here, which are used for the optical flow function
        """
        lucas_kanade = dict(winSize=(15, 15), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0, 255, (100, 3))
        if input_video.isOpened():
            ret, old_frame = input_video.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(old_frame)
            p0, points_coordinates_3d, advert_world, advert_2_world, shadow_1_coordinates = optical_flow_point_selector(
                video_name)

            """
            Point array is selectively copied by checking for Z axis values, as initial guess cannot have Z axis values
            """
            initial_image_points = []
            image_points = []
            initial_coordinates_3d = []
            for count, point in enumerate(p0):
                if points_coordinates_3d[0][count][2] == 0:
                    initial_image_points.append(point[0])
                    initial_coordinates_3d.append(points_coordinates_3d[0][count])
                image_points.append(point[0])

            """
            Additional points for camera calibration
            """
            if "tennis" in video_name:
                if video_name == "tennis_1.mp4":
                    initial_image_points.append(np.array([522, 781], dtype=np.float32))
                    initial_image_points.append(np.array([359, 777], dtype=np.float32))
                if video_name == "tennis_2.mp4":
                    initial_image_points.append(np.array([473, 815], dtype=np.float32))
                    initial_image_points.append(np.array([303, 817], dtype=np.float32))
                initial_coordinates_3d.append(np.array([9.601, 0, 0], dtype=np.float32))
                initial_coordinates_3d.append(np.array([10.973, 0, 0], dtype=np.float32))

            initial_image_points = np.vstack([[initial_image_points]])
            image_points = np.vstack([[image_points]])
            initial_coordinates_3d = np.vstack([[initial_coordinates_3d]])
            print(initial_image_points)
            print(initial_coordinates_3d)
            # raise SystemExit
            """
            Camera calibration is performed twice before launching the main loop,
            first calibration uses points that are located on 2D plane to generate initial guess for camera matrix
            second calibration includes 3D points and improves the initial guess
            """
            _, initialCameraMatrix, _, _, _ = cv2.calibrateCamera(initial_coordinates_3d,
                                                                  initial_image_points,
                                                                  old_gray.shape[::-1],
                                                                  None, None,
                                                                  flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

            _, CameraMatrix, dist, rotation_vec, translation_vec = cv2.calibrateCamera(points_coordinates_3d,
                                                                                       image_points,
                                                                                       old_gray.shape[::-1],
                                                                                       initialCameraMatrix, None,
                                                                                       flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_CB_NORMALIZE_IMAGE)
            print(CameraMatrix)
            rotation_vec = rotation_vec[0]
            translation_vec = translation_vec[0]
        else:
            print("Could not open video")
            raise SystemExit

        """ 
        Main video loop 
        """
        while frame_count < input_video.get(cv2.CAP_PROP_FRAME_COUNT):
            startTime = time.time()
            ret, frame = input_video.read()
            try:
                if ret:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    """
                    Optical flow is calculated using 2 frames and a set of key-points p0
                    """
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lucas_kanade)

                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        if SHOW_TRACKING_POINTS:
                            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

                    main_frame = cv2.add(frame, mask)

                    """
                    Optical flow points are remapped for solvePNP function
                    """
                    points = []
                    for point in p1:
                        points.append(point[0])
                    image_points = np.vstack([[points]])

                    _, rotation_vec, translation_vec = cv2.solvePnP(points_coordinates_3d, image_points, CameraMatrix,
                                                                    None)

                    advert_position, _ = cv2.projectPoints(advert_world, rotation_vec, translation_vec, CameraMatrix,
                                                           None)

                    advert_shadow_position, _ = cv2.projectPoints(shadow_1_coordinates, rotation_vec,
                                                                  translation_vec, CameraMatrix, None)

                    advert_2_position, _ = cv2.projectPoints(advert_2_world, rotation_vec, translation_vec,
                                                             CameraMatrix,
                                                             None)

                    """
                    To make working with points easier, the result from projectPoints is remapped
                    """
                    # Advert 1 points
                    advert_bot_left = (int(advert_position[0][0][0]), int(advert_position[0][0][1]))
                    advert_top_left = (int(advert_position[1][0][0]), int(advert_position[1][0][1]))
                    advert_top_right = (int(advert_position[2][0][0]), int(advert_position[2][0][1]))
                    advert_bot_right = (int(advert_position[3][0][0]), int(advert_position[3][0][1]))

                    # Advert 1 shadow points
                    shadow_bot_left = (int(advert_shadow_position[0][0][0]), int(advert_shadow_position[0][0][1]))
                    shadow_top_left = (int(advert_shadow_position[1][0][0]), int(advert_shadow_position[1][0][1]))
                    shadow_top_right = (int(advert_shadow_position[2][0][0]), int(advert_shadow_position[2][0][1]))
                    shadow_bot_right = (int(advert_shadow_position[3][0][0]), int(advert_shadow_position[3][0][1]))

                    # Advert 2 points
                    advert_2_bot_left = (int(advert_2_position[0][0][0]), int(advert_2_position[0][0][1]))
                    advert_2_top_left = (int(advert_2_position[1][0][0]), int(advert_2_position[1][0][1]))
                    advert_2_top_right = (int(advert_2_position[2][0][0]), int(advert_2_position[2][0][1]))
                    advert_2_bot_right = (int(advert_2_position[3][0][0]), int(advert_2_position[3][0][1]))

                    """
                    Axis line for measurements
                    cv2.line(main_frame, advert_bot_left[:2], advert_top_left[:2], (255, 100, 0), 2)
                    """

                    """
                    HSV masks are used to separate advertisement background from foreground
                    If advertisements are on different backgrounds, 2 separate masks are used
                    """

                    hsv_advert_mask_2 = None
                    if video_name == "basketball_1.mp4":
                        hsv_advert_mask = create_hsv_mask(hsv_filter_frame=frame,
                                                          hsv_low=[167, 75, 88], hsv_high=[178, 255, 163])
                        hsv_advert_mask_2 = create_hsv_mask(hsv_filter_frame=frame,
                                                            hsv_low=[170, 213, 83], hsv_high=[179, 255, 185])
                    elif video_name == "basketball_2.mp4":
                        hsv_advert_mask = create_hsv_mask(hsv_filter_frame=frame,
                                                          hsv_low=[158, 125, 82], hsv_high=[179, 255, 152])
                        hsv_advert_mask_2 = create_hsv_mask(hsv_filter_frame=frame,
                                                            hsv_low=[168, 213, 83], hsv_high=[179, 255, 185])
                    elif video_name == "tennis_1.mp4":
                        hsv_advert_mask = create_hsv_mask(hsv_filter_frame=frame,
                                                          hsv_low=[31, 78, 146], hsv_high=[48, 123, 202])
                        hsv_advert_mask_2 = create_hsv_mask(hsv_filter_frame=frame,
                                                            hsv_low=[129, 72, 140], hsv_high=[140, 114, 202])

                    elif video_name == "tennis_2.mp4":
                        hsv_advert_mask = create_hsv_mask(hsv_filter_frame=frame,
                                                          hsv_low=[19, 55, 149], hsv_high=[51, 108, 214])
                        hsv_advert_mask_2 = create_hsv_mask(hsv_filter_frame=frame,
                                                            hsv_low=[118, 55, 134], hsv_high=[130, 93, 163])
                    else:
                        hsv_advert_mask = create_hsv_mask(hsv_filter_frame=frame,
                                                          hsv_low=[0, 0, 0], hsv_high=[179, 255, 255])

                    aH, aW, c = advertisement.shape
                    advertPointMatrix = np.float32([[0, 0], [aW, 0], [0, aH], [aW, aH]])

                    """
                    Advert 1 Shadow perspective warp and masking to frame
                    """
                    if SHOW_ADVERTISEMENT_1:
                        shadow_opacity = -3.5  # dark < 0 < white,
                        advertLocationMatrix = np.float32(
                            [shadow_top_left[:2], shadow_top_right[:2], shadow_bot_left[:2], shadow_bot_right[:2]])
                        perspectiveMatrix = cv2.getPerspectiveTransform(advertPointMatrix, advertLocationMatrix)
                        shadowWarpResult = cv2.warpPerspective(shadow, perspectiveMatrix, (1920, 1080))
                        shadowWarpResult_blur = cv2.GaussianBlur(shadowWarpResult, (7, 7), 11)
                        masked_shadow = cv2.bitwise_and(shadowWarpResult_blur, shadowWarpResult_blur,
                                                        mask=hsv_advert_mask)
                        main_frame = cv2.addWeighted(main_frame, 1, masked_shadow, shadow_opacity, 0)

                    """
                    Advert 1 perspective warp and masking to frame
                    """
                    if SHOW_ADVERTISEMENT_1:
                        advertLocationMatrix = np.float32(
                            [advert_top_left[:2], advert_top_right[:2], advert_bot_left[:2], advert_bot_right[:2]])
                        perspectiveMatrix = cv2.getPerspectiveTransform(advertPointMatrix, advertLocationMatrix)
                        advertWarpResult = cv2.warpPerspective(advertisement, perspectiveMatrix, (1920, 1080))
                        masked_advert_frame = cv2.bitwise_and(advertWarpResult, advertWarpResult, mask=hsv_advert_mask)
                        main_frame = cv2.add(main_frame, masked_advert_frame)

                    """
                    Advert 2 perspective warp and masking to frame
                    """
                    if SHOW_ADVERTISEMENT_2:
                        advert_2_LocationMatrix = np.float32(
                            [advert_2_top_left[:2], advert_2_top_right[:2], advert_2_bot_left[:2],
                             advert_2_bot_right[:2]])
                        perspectiveMatrix_2 = cv2.getPerspectiveTransform(advertPointMatrix, advert_2_LocationMatrix)
                        advertWarpResult_2 = cv2.warpPerspective(advertisement, perspectiveMatrix_2, (1920, 1080))
                        if hsv_advert_mask_2 is not None:
                            hsv_advert_mask = hsv_advert_mask_2
                        masked_advert_frame = cv2.bitwise_and(advertWarpResult_2, advertWarpResult_2,
                                                              mask=hsv_advert_mask)
                        main_frame = cv2.add(main_frame, masked_advert_frame)

                    """
                    For debugging purposes:
                    ESC: exits the program
                    SPACEBAR: pauses the playback
                    """
                    k = cv2.waitKey(1)
                    if k == 27:
                        cv2.destroyAllWindows()
                        input_video.release()
                        raise SystemExit
                    if k == 32:
                        cv2.waitKey(0)
                    """
                    For showing the outputs, a function is used that creates multiple output windows according to input
                    """
                    show_multiple_output([main_frame], 2)
                    if SAVE_VIDEO:
                        videowriter.write(main_frame)

                    """
                    Current frame is copied to old frame for optical flow
                    """
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
            except Exception as error:
                traceback.print_exc()
                print(error)
            frame_count += 1
            """
            Measure execution time
            """
            executionTime = (time.time() - startTime)
            print("Execution time in ms: " + str(executionTime * 1000))

        k = cv2.waitKey(0)  # Disable video autoplay each loop
        if k == 27:
            if SAVE_VIDEO:
                videowriter.release()
            cv2.destroyAllWindows()
            input_video.release()
            raise SystemExit
