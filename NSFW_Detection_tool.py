import cv2
import numpy as np
from PIL import Image
from nudenet import NudeDetector
import mss 
import time
import os
import matplotlib.pyplot as plt

classifier = NudeDetector()

sct = mss.mss()
monitor = sct.monitors[1]

def detect_nsfw_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img =Image.fromarray(rgb)
    img.save("temp_screen.jpg")
    result = classifier.detect("temp_screen.jpg")
    os.remove("temp_screen.jpg")

    is_unsafe = len(result) > 0
    confidence =result[0]['score'] if is_unsafe else 0.0

    label = "unsafe" if is_unsafe else "safe"

    return {"class": label, "confidence": confidence}

print("NSFW screen Monitoring STARTED")

try:
    while True:
        screenShot = sct.grab(monitor)
        frame = np.array(screenShot)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detect_nsfw_from_frame(frame)
        label = result['class']
        confidence = result['confidence']

        print(f'{label.upper()} ({confidence:.2f})')

        if label == "unsafe":
            print("NSFW Content Detected With confidence {confidence:.2f}")

            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"NSFW Monitor: {label.upper()} ({confidence:.2f})")
            plt.axis('off')
            plt.pause(0.01)
            plt.clf()

            time.sleep(1)

except KeyboardInterrupt:
    print("stoppes by user")
finally:
    plt.close()
