# DAmageNet
DAmageNet is a dataset of 50000 universal adversarial examples constructed from ImageNet validation set. We test the DAmageNet in the zero-shot style. We measure its ability to fool various organism/object classifiers and compare this to the original ImageNet validation baseline.

## Tested models evaluation procedure
We test Resnet18 and Resnet-50 pretrained models from torchvision.models package. The only problem is that those models are not binary classifiers, but instead they assign each input image one category $c$ beloning to the set of 1000 ImageNet categories $C$. We solve this by mapping each of the 1000 categories to only two categories - $(\text{organism}, \text{object})$. Let's call this mapping $M(c)$, where $c$ is an ImageNet category. This mapping is done by taking the corresponding wordnet synset and checking whether organism synset is its hypernym. Final classification $M(c_{best})$ is done by taking the argmax over all 1000 categories - $c_{best} = \underset{c \in C}{\argmax} f_c(x)$.

### Evaluated ImageNet validation subset
The idea is to take an image containing a living thing, an organism, and producing an adversarial example, which fools the target model into thinking there isn't anything living in the picture. 
Therefore we select 20500 organism images from the whole ImageNet validation set and evaluate only this subset.

## Imagenet validation baseline
| Model     | Fool rate |
| --------- | --------- |
| Resnet-18 | 2.03%     |
| Resnet-50 | 1.72%     |
| Gvision   | 31.95%    |

## DAmageNet
| Model     | Fool rate |
| --------- | --------- |
| Resnet-18 | 15.90%    |
| Resnet-50 | 16.03%    |
| Gvision   | 32.05%    |

Gvision is only tested on the first 2000 images, because queries are expensive.
As we can see, Gvision is totally robust to DAmageNet.
