""" Test HSV filters on a live camera with brightness/contrast controls """

import cv2 as cv
import numpy as np
import json
import os
from time import sleep

LOW = np.array([0, 0, 0])
UPP = np.array([180, 255, 255])

FILTER_WINDOW = "FILTER MARKERS"
SOURCE_WINDOW = "src1"
FILTERED_WINDOW = "FILTER"

BRIGHTNESS_MIN = -40
BRIGHTNESS_MAX = 40

CONTRAST_MIN = 0.0
CONTRAST_MAX = 2.0
CONTRAST_SCALE = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)


def noop(_):
    pass


def min_hue(value):
    LOW[0] = value


def min_sat(value):
    LOW[1] = value


def min_bri(value):
    LOW[2] = value


def max_hue(value):
    UPP[0] = value


def max_sat(value):
    UPP[1] = value


def max_bri(value):
    UPP[2] = value


def get_brightness():
    slider_value = cv.getTrackbarPos("BRIGHTNESS (-40 to 40)", FILTER_WINDOW)
    return slider_value + BRIGHTNESS_MIN


def get_contrast():
    slider_value = cv.getTrackbarPos("CONTRAST (0.00 to 2.00)", FILTER_WINDOW)
    return slider_value / CONTRAST_SCALE


def apply_contrast_brightness(frame, contrast, brightness):
    return cv.convertScaleAbs(
        frame,
        alpha=contrast,
        beta=brightness,
    )


cv.namedWindow(FILTER_WINDOW)

cv.createTrackbar("MIN_HUE", FILTER_WINDOW, 0, 180, min_hue)
cv.createTrackbar("MIN_SAT", FILTER_WINDOW, 0, 255, min_sat)
cv.createTrackbar("MIN_BRI", FILTER_WINDOW, 0, 255, min_bri)

cv.createTrackbar("MAX_HUE", FILTER_WINDOW, 180, 180, max_hue)
cv.createTrackbar("MAX_SAT", FILTER_WINDOW, 255, 255, max_sat)
cv.createTrackbar("MAX_BRI", FILTER_WINDOW, 255, 255, max_bri)

cv.createTrackbar(
    "BRIGHTNESS (-40 to 40)",
    FILTER_WINDOW,
    40,
    BRIGHTNESS_MAX - BRIGHTNESS_MIN,
    noop,
)

cv.createTrackbar(
    "CONTRAST (0.00 to 2.00)",
    FILTER_WINDOW,
    100,
    int(CONTRAST_MAX * CONTRAST_SCALE),
    noop,
)

camera_index = int(config.get("camera_index", 0))

cap = cv.VideoCapture(camera_index)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while not cap.isOpened():
    cap = cv.VideoCapture(camera_index)
    print("Waiting for camera...")
    sleep(0.05)

while True:
    ret, src = cap.read()
    if not ret:
        break

    brightness = get_brightness()
    contrast = get_contrast()

    adjusted = apply_contrast_brightness(src, contrast, brightness)

    hsv = cv.cvtColor(adjusted, cv.COLOR_BGR2HSV)
    msk = cv.inRange(hsv, LOW, UPP)
    filtered = cv.bitwise_and(adjusted, adjusted, mask=msk)

    preview = adjusted.copy()

    cv.putText(
        preview,
        f"contrast: {contrast:.2f} | brightness: {brightness}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv.putText(
        preview,
        f"LOW: {LOW.tolist()} | HIGH: {UPP.tolist()}",
        (20, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv.imshow(SOURCE_WINDOW, preview)
    cv.imshow(FILTERED_WINDOW, filtered)

    key = cv.waitKey(1)

    if key == ord("p"):
        print_config = {
            "contrast": round(contrast, 2),
            "brightness": brightness,
            "low": LOW.tolist(),
            "high": UPP.tolist(),
        }
        print(print_config)

    if key == ord("b"):
        print("Exit...")
        break

cap.release()
cv.destroyAllWindows()