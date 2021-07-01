from advpipe.blackbox.cloud import GVisionBlackBox
from advpipe import utils
from munch import Munch
import sys
import numpy as np

# Load and classify images given as argmuments

gvision = GVisionBlackBox(Munch.fromDict({"name":"gvision"}))

for img_path in sys.argv[1:]:
    print("Img:", img_path)

    np_img = utils.load_image_to_numpy(img_path)
    cloud_labels = gvision.classify(np_img)

    pil_img = utils.convert_to_pillow(np_img)
    img_with_descs = utils.write_text_to_img(pil_img, str(cloud_labels))

    utils.show_img(np.array(img_with_descs))


