from enum import Enum
import cv2
import argparse
import os.path as pth
import numpy as np
import matplotlib.pyplot as plt

"""
Gets image path , get edges , then save it 
"""


class BLUR_TYPE(Enum):
    GAUSSIAN_BLUR = 1,
    MEDIAN_BLUR = 2,
    AVERGAE_BLUR = 3,


def GetBlur(image_content, blurStr):
    if blurStr == "GAUSSIAN":
        return cv2.GaussianBlur(image_content, (5, 5), 500)
    elif blurStr == "MEDIAN":
        return cv2.medianBlur(src=image_content, ksize=5)
    else:
        return cv2.filter2D(image_content, -1, np.ones((5, 5), np.float32)/25)


def LoadArgs():
    ap = argparse.ArgumentParser(
        prog="Sketch", description="Creates sketch for the image", exit_on_error=True, add_help=True)
    ap.add_argument("-i", "--image-path", help="Image path", required=True)
    ap.add_argument("-o", "--output-path", help="Output path", required=False)
    ap.add_argument("-b", "--blur", help="Blur Type", required=False)

    return vars(ap.parse_args())


if __name__ == "__main__":
    args = LoadArgs()

    image_path = args["image_path"]
    image_name = pth.basename(image_path)
    output_path = "Sketch_{0}".format(image_name)
    blurStr = "GAUSSIAN"
    image_content = ""

    if not args["output_path"] == None:
        output_path = args["output_path"]
    if not args["blur"] == None:
        blurStr = args["blur"]

    if image_path == "camera":
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            isp, frame = cap.read()
            if not isp:
                raise NotImplementedError
            else:
                image_content = frame
                image_path = "{0}.png".format("camera")
                image_name = "camera"
                output_path = "Sketch_camera.png"
    else:
        image_content = cv2.imread(filename=image_path, flags=1)

    # cv2.imshow(mat=image_content, winname="{0}".format(image_name))

    # if cv2.waitKey(1000 * 2) == ord("q"):
    #     print("show image quitted")

    """
    Blur the image
    apply canny to get edges 
    threshold to get darker images
    """

    blur_content = GetBlur(image_content=image_content, blurStr=blurStr)
    image_content1 = cv2.cvtColor(src=image_content, code=cv2.COLOR_BGR2RGB)
    blur_content1 = cv2.cvtColor(src=blur_content, code=cv2.COLOR_BGR2RGB)

    plt.title("Images")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

    ax1.set_title("Original Content")
    ax1.imshow(image_content1)

    ax2.set_title("Blurred Content")
    ax2.imshow(blur_content1)

    fig.show(True)
    cv2.waitKey(1000 * 2)
    fig.set_visible(False)
    cv2.destroyAllWindows()

    edge_content = cv2.Canny(image=blur_content, threshold1=50, threshold2=150)
    cv2.imshow(mat=edge_content, winname="edge content")
    cv2.waitKey(1000 * 2)
    print(output_path)
    cv2.imwrite(filename=output_path, img=edge_content)
