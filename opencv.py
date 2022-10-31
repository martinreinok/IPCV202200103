import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Program modifiers
Trackbars = True


class Video:
    FPS = None
    FRAME_COUNT = None

    class SIZE:
        """X, Y (pixel)"""
        X = None
        Y = None

    def read(self, filename):
        input_video = cv2.VideoCapture(filename)
        if not input_video.isOpened():
            print("Error opening the video file")
        else:
            self.SIZE.X = int(input_video.get(3))
            self.SIZE.Y = int(input_video.get(4))
            self.FPS = int(input_video.get(5))
            self.FRAME_COUNT = int(input_video.get(7))
            return input_video

    def show_multiple_output(self, video_feeds_list, scale):
        for i, video_feed in enumerate(video_feeds_list):
            cv2.namedWindow(f"Output {i}", cv2.WINDOW_KEEPRATIO)
            cv2.imshow(f"Output {i}", video_feed)
            cv2.resizeWindow(f"Output {i}", Video.SIZE.X // scale, Video.SIZE.Y // scale)


class Advert:

    def write_trackbar_file(self):
        data = [f"hue_low = {self.trackbar_value().hue_low}\n"
                f"hue_high = {self.trackbar_value().hue_high}\n"
                f"sat_low = {self.trackbar_value().sat_low}\n"
                f"sat_high = {self.trackbar_value().sat_high}\n"
                f"value_low = {self.trackbar_value().value_low}\n"
                f"value_high = {self.trackbar_value().value_high}\n"
                f"canny_1 = {self.trackbar_value().canny_1}\n"
                f"canny_2 = {self.trackbar_value().canny_2}\n"
                f"morph_1 = {self.trackbar_value().morph_1}\n"
                f"morph_2 = {self.trackbar_value().morph_1}\n"
                f"kerdialate_1 = {self.trackbar_value().kerdialate_1}\n"
                f"kerdialate_2 = {self.trackbar_value().kerdialate_2}\n"
                f"dialateItt = {self.trackbar_value().dialateItt}\n"
                f"kererode_1 = {self.trackbar_value().kererode_1}\n"
                f"kererode_2 = {self.trackbar_value().kererode_1}\n"
                f"erodeItt = {self.trackbar_value().erodeItt}\n"
                f"corner_1 = {self.trackbar_value().corner_1}\n"
                f"corner_2 = {self.trackbar_value().corner_2}\n"
                f"line_rho = {self.trackbar_value().rho}\n"
                f"line_threshold = {self.trackbar_value().threshold}\n"
                f"line_min = {self.trackbar_value().min_line_length}\n"
                f"line_gap = {self.trackbar_value().max_line_gap}\n"]
        with open("trackbar_memory", "w+") as memory:
            memory.writelines(data)

    def read_trackbar_file(self):
        data = [int(x.strip().split(" ")[2]) for x in open("trackbar_memory")]
        return data

    def convert_hsv(self, input_video):
        hsv = cv2.cvtColor(input_video, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([self.trackbar_value().hue_low, self.trackbar_value().sat_low, self.trackbar_value().value_low])
        higher_hsv = np.array([self.trackbar_value().hue_high, self.trackbar_value().sat_high, self.trackbar_value().value_high])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        hsv_frame = cv2.bitwise_and(input_video, input_video, mask=mask)
        return hsv_frame

    def nothing(self, x):
        pass

    def trackbars(self):
        cv2.namedWindow("Trackbars", 2)
        cv2.resizeWindow("Trackbars", 550, 10)
        memory = self.read_trackbar_file()
        # HSV
        cv2.createTrackbar("hue_low", "Trackbars", memory[0], 179, self.nothing)
        cv2.createTrackbar("hue_high", "Trackbars", memory[1], 179, self.nothing)
        cv2.createTrackbar("sat_low", "Trackbars", memory[2], 255, self.nothing)
        cv2.createTrackbar("sat_high", "Trackbars", memory[3], 255, self.nothing)
        cv2.createTrackbar("value_low", "Trackbars", memory[4], 255, self.nothing)
        cv2.createTrackbar("value_high", "Trackbars", memory[5], 255, self.nothing)
        # CANNY
        cv2.createTrackbar("canny_1", "Trackbars", memory[6], 500, self.nothing)
        cv2.createTrackbar("canny_2", "Trackbars", memory[7], 500, self.nothing)
        # Morph
        cv2.createTrackbar("morph_1", "Trackbars", memory[8], 100, self.nothing)
        cv2.createTrackbar("morph_2", "Trackbars", memory[9], 100, self.nothing)
        cv2.createTrackbar("kerdialate_1", "Trackbars", memory[10], 50, self.nothing)
        cv2.createTrackbar("kerdialate_2", "Trackbars", memory[11], 50, self.nothing)
        cv2.createTrackbar("dialateItt", "Trackbars", memory[12], 50, self.nothing)
        cv2.createTrackbar("kererode_1", "Trackbars", memory[13], 50, self.nothing)
        cv2.createTrackbar("kererode_2", "Trackbars", memory[14], 50, self.nothing)
        cv2.createTrackbar("erodeItt", "Trackbars", memory[15], 50, self.nothing)
        # THRESHOLD
        cv2.createTrackbar("corner_1", "Trackbars", memory[16], 100, self.nothing)
        cv2.createTrackbar("corner_2", "Trackbars", memory[17], 500, self.nothing)
        cv2.createTrackbar("harris_bSize", "Trackbars", 1, 25, self.nothing)
        cv2.createTrackbar("harris_kSize", "Trackbars", 7, 10, self.nothing)
        cv2.createTrackbar("harris_k", "Trackbars", 1, 1000, self.nothing)
        # LINE DETECTION
        cv2.createTrackbar("line_rho", "Trackbars", memory[18], 179, self.nothing)
        cv2.createTrackbar("line_threshold", "Trackbars", memory[19], 1000, self.nothing)
        cv2.createTrackbar("line_min", "Trackbars", memory[20], 1000, self.nothing)
        cv2.createTrackbar("line_gap", "Trackbars", memory[21], 10000, self.nothing)

    def trackbar_value(self):
        class Values:
            if Trackbars:
                # Hue
                hue_low = cv2.getTrackbarPos("hue_low", "Trackbars")
                hue_high = cv2.getTrackbarPos("hue_high", "Trackbars")
                sat_low = cv2.getTrackbarPos("sat_low", "Trackbars")
                sat_high = cv2.getTrackbarPos("sat_high", "Trackbars")
                value_low = cv2.getTrackbarPos("value_low", "Trackbars")
                value_high = cv2.getTrackbarPos("value_high", "Trackbars")
                # Canny
                canny_1 = cv2.getTrackbarPos("canny_1", "Trackbars")
                canny_2 = cv2.getTrackbarPos("canny_2", "Trackbars")
                #morph operators
                morph_1 = cv2.getTrackbarPos("morph_1", "Trackbars")
                morph_2 = cv2.getTrackbarPos("morph_2", "Trackbars")
                kerdialate_1 = cv2.getTrackbarPos("kerdialate_1", "Trackbars")
                kerdialate_2 = cv2.getTrackbarPos("kerdialate_2", "Trackbars")
                dialateItt = cv2.getTrackbarPos("dialateItt", "Trackbars")
                kererode_1 = cv2.getTrackbarPos("kererode_1", "Trackbars")
                kererode_2 = cv2.getTrackbarPos("kererode_2", "Trackbars")
                erodeItt = cv2.getTrackbarPos("erodeItt", "Trackbars")
                # Corner
                corner_1 = cv2.getTrackbarPos("corner_1", "Trackbars")
                corner_2 = cv2.getTrackbarPos("corner_2", "Trackbars")
                harris_bSize = cv2.getTrackbarPos("harris_bSize", "Trackbars")
                harris_kSize = cv2.getTrackbarPos("harris_kSize", "Trackbars")
                harris_k = cv2.getTrackbarPos("harris_k", "Trackbars")
                # Lines
                rho = cv2.getTrackbarPos("line_rho", "Trackbars")
                threshold = cv2.getTrackbarPos("line_threshold", "Trackbars")
                min_line_length = cv2.getTrackbarPos("line_min", "Trackbars")
                max_line_gap = cv2.getTrackbarPos("line_gap", "Trackbars")
            else:
                # Hue
                hue_low = 170  # hue low
                hue_high = 175  # hue high
                sat_low = 150  # saturation low
                sat_high = 235  # saturation high
                value_low = 91  # value low
                value_high = 178  # value high
                # Canny
                canny_1 = 58  # canny 1
                canny_2 = 109  # canny 2
                # Morph
                morph_1 = 50
                morph_1 = 50
                kerdialate_1 = 1
                kerdialate_2 = 1
                dialateItt = 1
                kererode_1 = 1
                kererode_2 = 1
                erodeItt = 1
                # Corner
                corner_1 = 49  # corner 1
                corner_2 = 291 # corner 2
                # Lines
                rho = 97  # distance resolution in pixels of the Hough grid
                threshold = 85  # minimum number of votes (intersections in Hough grid cell)
                min_line_length = 0  # minimum number of pixels making up a line
                max_line_gap = 109  # maximum gap in pixels between connectable line segments
        return Values

    def detect_lines(self, canny_edge_detection):
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        line_image = np.copy(canny_edge_detection) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = None
        try:
            lines = cv2.HoughLinesP(canny_edge_detection, self.trackbar_value().rho/10, theta, self.trackbar_value().threshold,
                                    np.array([]), self.trackbar_value().min_line_length, self.trackbar_value().max_line_gap)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        except Exception as error:
            print(error)
            pass
        return line_image, lines

    def draw_line_intersections(self, lines, output_image):
        try:
            for line1 in lines:
                x1, y1, x2, y2 = line1[0]
                for line2 in lines:
                    x3, y3, x4, y4 = line2[0]
                    if x1 == x3 and y1 == y3:
                        pass
                    else:
                        u = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (
                                (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                        x = int(x1 + u * (x2 - x1))
                        y = int(y1 + u * (y2 - y1))
                        cv2.circle(output_image, (x, y), 4, (255, 0, 0), 3)
        except Exception as e:
            print(e)
            pass


Video = Video()
Advert = Advert()
video = None
gaussian_blur = (11, 11)

if __name__ == "__main__":
    video = Video.read("videos\Tennis_video_1.mp4")
    if Trackbars:
        Advert.trackbars()
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            black_screen = cv2.cvtColor(np.copy(frame) * 0, cv2.COLOR_BGR2GRAY)  # creating a blank to draw anything

            hsv_frame = Advert.convert_hsv(frame)
            gray_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2GRAY)

            # contours, hierarchy = cv2.findContours(canny_morphErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contourSizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]
            maxContour = max(contourSizes, key=lambda x: x[0])[1]
            cv2.drawContours(black_screen, [maxContour], -1, (255, 0, 0), -1 )
            contourKernal = np.ones((5,5))
            maxContourDialate = cv2.dilate(black_screen, contourKernal, iterations= 6)

            ROIimg = cv2.bitwise_and(gray_frame, maxContourDialate)


            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(ROIimg, mask=None, **feature_params)
            gray_blur_frame = cv2.GaussianBlur(ROIimg, gaussian_blur, 0)
            gray_blur_frame_morph = cv2.morphologyEx(ROIimg, cv2.MORPH_GRADIENT, np.ones((1, 1)))
            hsvmorph = cv2.morphologyEx(ROIimg, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_RECT, (Advert.trackbar_value().morph_1, Advert.trackbar_value().morph_2)))
            canny_edges = cv2.Canny(hsvmorph, Advert.trackbar_value().canny_1, Advert.trackbar_value().canny_2)
            
            kernelDial = np.ones((Advert.trackbar_value().kerdialate_1, Advert.trackbar_value().kerdialate_2))
            canny_morphDial = cv2.dilate(canny_edges, kernelDial, iterations = Advert.trackbar_value().dialateItt)

            kernelErode = np.ones((Advert.trackbar_value().kererode_1, Advert.trackbar_value().kererode_2))
            canny_morphErode = cv2.erode(canny_morphDial,kernelErode, iterations = Advert.trackbar_value().erodeItt)

            detected_lines, lines = Advert.detect_lines(canny_morphErode)
            print(lines)

            corners = cv2.cornerHarris(detected_lines,Advert.trackbar_value().harris_bSize,Advert.trackbar_value().harris_kSize,Advert.trackbar_value().harris_k / 1000)
            cornersDial = cv2.dilate(corners, np.zeros((7,7),np.uint8), iterations=2)
            frame[cornersDial>0.01*cornersDial.max()]=[0,0,255]

            Advert.draw_line_intersections(lines, frame)
            Video.show_multiple_output([frame, hsv_frame, detected_lines, canny_edges, canny_morphErode, ROIimg], 1)
            # cv2.waitKey(0)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.waitKey(int(Video.FPS)) & 0xFF == ord('q'):
            break

video.release()
if Trackbars:
    Advert.write_trackbar_file()
cv2.destroyAllWindows()
