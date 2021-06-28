from __future__ import annotations
from os import path
from typing import Sequence, Tuple, Any
from munch import Munch, unmunchify
import inspect
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image, ImageDraw
import cv2
from advpipe.log import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Optional


def scale_img(img: Image, target_size: int) -> Image:
    """Re-scale image while preserving its aspect ratio"""
    x, y = img.size

    # the shorter side will be scaled to the target_size
    if x < y:
        scale_ratio = target_size / x
    else:
        scale_ratio = target_size / y

    return img.resize((int(x * scale_ratio), int(y * scale_ratio)))


def mkdir_p(dir_path: str) -> None:
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def show_img(np_img: np.ndarray, method: str = "pyplot") -> None:
    if method == "PIL":
        convert_to_pillow(np_img).show()
    elif method == "opencv":
        cv2.imshow("", np_img[..., ::-1])
        cv2.waitKey(0)
    elif method == "pyplot":
        plt.imshow(np_img, interpolation="bicubic")
        plt.show()
    else:
        raise ValueError("utils.show_img: Unsupported method")


def is_img_filename(img_fn: str) -> bool:
    extensions = [".png", ".jpg", ".jpeg"]
    img_fn = img_fn.lower()
    return any([img_fn.endswith(ext) for ext in extensions])


def load_image_to_numpy(img_path: str) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    # convert image to numpy array
    np_img = np.asarray(image, dtype=np.float32) / 255
    return np_img


def write_text_to_img(img: Image, text: str, max_lines: int = 20) -> Image:
    """Writes text to the top of the image"""

    font_size = 10
    text = "\n".join(text.split("\n")[:max_lines])
    n_lines = len(text.split("\n"))
    margin = int(n_lines * font_size * 1.5)

    width, height = img.size
    new_img = Image.new("RGB", size=(width + 200, max(height, margin)))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.text((width + 10, 0), text, (255, 255, 255))

    return new_img


def convert_to_pillow(np_img: np.ndarray) -> Image:
    if np_img.dtype != np.uint8:
        if np_img.min() >= 0 and np_img.max() <= 1:
            np_img = np.asarray(np_img * 255, dtype=np.uint8)
        elif np_img.min() >= 0 and np_img.max() <= 255:
            np_img = np.asarray(np_img, dtype=np.uint8)
        else:
            raise ValueError("Cannot convert numpy array to PIL image: unsupported image format")

    return Image.fromarray(np_img)


def labels_and_scores_to_str(labels_and_scores: Sequence[Tuple[str, float]]) -> str:
    """Convert labels and scores tuple-list to pretty-printable string"""
    return "\n".join(list(map(lambda l_s: l_s[0] + ": " + str(l_s[1]), labels_and_scores)))


# deprecated
def clip_linf(orig_img: np.ndarray, pertubed_img: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    min_boundary = np.clip(orig_img - epsilon * np.ones_like(orig_img), 0, 1)
    max_boundary = np.clip(orig_img + epsilon * np.ones_like(orig_img), 0, 1)
    return np.clip(pertubed_img, min_boundary, max_boundary)


def load_yaml(yaml_filename: str) -> Munch:
    with open(yaml_filename, 'r') as stream:
        return Munch.fromDict(yaml.safe_load(stream))


# TODO: this is quite a bad fuction name
def rel_to_abs_path(relative_path: str) -> str:
    """convert caller's relative path to absolute path"""
    callers_path = inspect.stack()[1].filename
    return path.normpath(path.join(path.dirname(path.abspath(callers_path)), relative_path))


def convert_to_absolute_path(module_relative_path: str) -> str:
    """Convert path relative to advpipe's module root directory to its absolute variant"""
    return path.join(get_abs_module_path(), module_relative_path)


def get_abs_module_path() -> str:
    return rel_to_abs_path(".")


class MaxFunctionCallsExceededException(Exception):
    pass


class LossCallCounter:
    def __init__(self, loss_fn: Callable[[np.ndarray], float], max_calls: int):
        self.loss_fn: Callable[[np.ndarray], float] = loss_fn
        self.last_loss_val: float = np.inf
        self.last_img: Optional[np.ndarray] = None
        self.max_calls: int = max_calls    # test comment
        self.i = 0

    def __call__(self, pertubed_image: np.ndarray) -> float:
        if self.i >= self.max_calls:
            msg = f"Max number of function calls exceeded (max_calls={self.max_calls})"
            logger.info(f"LossCallCounter: {msg}")
            raise MaxFunctionCallsExceededException(msg)

        self.i += 1
        self.last_loss_val = self.loss_fn(pertubed_image)
        self.last_img = pertubed_image
        return self.last_loss_val


def get_config_attr(conf: Munch, attr_name: str, default_val: Any) -> Any:
    """Returns config attribute value if it exists, otherwise returns default value"""
    val = default_val
    try:
        val = conf.__getattr__(attr_name)
    except AttributeError:
        pass
    return val


def serialize_config(conf: Munch) -> str:
    unmunched = unmunchify(conf)
    return yaml.dump(unmunched) # type: ignore