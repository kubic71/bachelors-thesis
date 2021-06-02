import logging
import logging.config
from os import path
from advpipe import utils
import imageio
import time

class CustomLogger:
    """Wrapper around standard python logger"""

    def __init__(self, logger, img_log_path="collected_cloud_data"):
        self.logger = logger
        self.img_log_path = img_log_path

    def __getattr__(self, attrname):
        return getattr(self.logger, attrname)

    def save_img(self, np_img, cloud_name, img_desc=""):
        dest = path.join(path.join(utils.get_abs_module_path(), self.img_log_path), cloud_name)

        # save img
        timestamp = time.time_ns()
        img_filename = f"{timestamp}_{cloud_name}_{img_desc}.png"
        img_full_path = path.join(dest, img_filename)
        imageio.imwrite(img_full_path, np_img)

        return img_full_path

    def save_cloud_labels(self, classification_labels, img_filename):
        """
        args: 
            classification_labels: list of tuples (label, score)
            img_filename as returned by save_img method
        """
        
        # remove .png file extension
        labels_filename = img_filename[:-4] + ".txt"

        with open(labels_filename, "w") as f:
            for label, score in classification_labels:
                f.write(f"{label};{score}\n")

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)

# create logger
logger = CustomLogger(logging.getLogger('root'))
