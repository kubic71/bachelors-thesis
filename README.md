# Exploring the vulnerabilities of commercial AI systems against adversarial attacks
My bachelor's TeX thesis and other related materials.

## AdvPipe
I set out to explore current state-of-the-art black-box adversarial attacks and how they fare in practical attack scenarios. I want to test out adversarial robustness of some commercial MLaaS cloud services like Google Vision API, Amazon Rekognition, Clarifai, Microsoft Azure AI etc.

Different black-box AI models have different APIs, and so do different attack algorithms. On top of that, I have some ideas how to tweak/change current attacks to better suit the APIs of different MLaaS threat models. AdvPipe is intended to be a modular pipeline, that would incorporate various attack regimes, target models and attack algorithms into single framework.

More on this [here](adv-pipe.md)


## Setup
- Tested on cuda 11.3, python 3.8
```
$ poetry install
```
That's all, really :) 


## *deprecated*
### Installing tensorflow-gpu and pytorch-gpu in the same conda environment
I had to specify 'pytorch' channel, otherwise conda always installs the cpu pytorch version 
```
$ conda create -n advpipe python=3.7 
$ conda install pytorch torchvision cudatoolkit tensorflow-gpu -c pytorch
```

### Language model
Nltk python library for wordnet label hypernyms

```
$ conda install -c anaconda nltk 
```

Huggingface transformers for zero-shot classification in case wordnet doesn't know the exact label phrase

```
$ pip install transformers  
```

### Other dependencies and packages installed throught the development process
Not all packages are required dependencies.
```
$ conda install pytest pyyaml munch pillow scikit-image matplotlib seaborn 
$ pip install google-cloud-vision opencv-python jedi neovim seaborn pre-commit eagerpy sklearn efficientnet_pytorch 


# python3.7-ported version of ebcic
$ git clone https://github.com/kubic71/ebcic
$ pip install -e ebcic
```



## Attacks

To get a sense of what this complicated title means in practice, checkout some of my other repos with [PoC](https://en.wikipedia.org/wiki/Proof_of_concept) adversarial attacks on Google Vision API:
- [Square attack](https://github.com/kubic71/square-attack)
- [TREMBA](https://github.com/kubic71/TREMBA)
- [RayS](https://github.com/kubic71/RayS)
- [Sparse-rs](https://github.com/kubic71/sparse-rs)


