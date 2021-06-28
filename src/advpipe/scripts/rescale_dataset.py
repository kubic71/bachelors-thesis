from advpipe import utils
import os
from PIL import Image

dataset_path = utils.get_abs_module_path() + "/datasets/imagenet_val"
output_dir = utils.get_abs_module_path() + "/datasets/imagenet_val_256_rescaled"

size = 256

utils.mkdir_p(output_dir)

for img_fn in os.listdir(dataset_path):
    if not utils.is_img_filename(img_fn):
        continue
    print(f"Re-scaling {img_fn}")

    img = Image.open(dataset_path + "/" + img_fn)
    new_img = utils.scale_img(img, size)
    new_img.save(output_dir + "/" + img_fn)







