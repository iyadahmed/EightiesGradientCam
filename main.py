import platform
import cv2
import numpy as np
from math import floor

WINDOW_NAME = "Camera"
STEP_SIZE = 10  # The height of each row in pixels


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

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply color map
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # Discretize image
        frame = frame[0::STEP_SIZE, :].repeat(STEP_SIZE, axis=0)

        # Show image
        cv2.imshow(WINDOW_NAME, frame)

        # Image will not show without calling waitKey to process window events
        cv2.waitKey(1)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
