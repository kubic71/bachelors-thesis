from advpipe import utils
from torchvision import transforms
import os
from PIL import Image

dataset_path = utils.get_abs_module_path() + "/datasets/imagenet_val"
output_dir = utils.get_abs_module_path() + "/datasets/imagenet_val_256_rescaled_224_center_cropped"

# size = 256

t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

utils.mkdir_p(output_dir)

for img_fn in os.listdir(dataset_path):
    if not utils.is_img_filename(img_fn):
        continue
    print(f"Re-scaling {img_fn}")

    img = Image.open(dataset_path + "/" + img_fn).convert("RGB")
    new_img = t(img)
    img_fn = img_fn.split('.')[0] + ".png"
    # new_img = utils.scale_img(img, size)
    new_img.save(output_dir + "/" + img_fn)







