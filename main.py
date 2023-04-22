import platform
import cv2
import numpy as np
from math import floor

WINDOW_NAME = "Camera"
STEP_SIZE = 30  # The height of each row in pixels


def main():
    cap = cv2.VideoCapture(
        0, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    )
    if not cap.isOpened():
        print("Failed to open video capture")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.Canny(frame, 100, 1000, apertureSize=7, L2gradient=False)

        # Denoise https://docs.opencv.org/4.7.0/d5/d69/tutorial_py_non_local_means.html
        # frame = cv2.fastNlMeansDenoising(frame)

        frame = frame.astype("f")

        # frame = cv2.blur(frame, (10, 10))
        # frame = cv2.GaussianBlur(frame, (15, 15), 0)
        frame *= 0.01

        # Discretize image
        # frame = frame[0::STEP_SIZE, :].repeat(STEP_SIZE, axis=0)

        # Apply color map
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # Show image
        cv2.imshow(WINDOW_NAME, frame)

        # Image will not show without calling waitKey to process window events
        cv2.waitKey(1)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
