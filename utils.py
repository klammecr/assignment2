from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def annotate(impath):
    im = Image.open(impath)
    from PIL import ImageOps
    im = ImageOps.exif_transpose(im)
    im = np.array(im)

    clicks = []

    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y, 1.])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    return clicks