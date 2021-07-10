# Exploring the vulnerabilities of commercial AI systems against adversarial attacks
My bachelor's TeX thesis and other related materials.

## AdvPipe
I set out to explore current state-of-the-art black-box adversarial attacks and how they fare in practical attack scenarios. I want to test out adversarial robustness of some commercial MLaaS cloud services like Google Vision API, Amazon Rekognition, Clarifai, Microsoft Azure AI etc.

Different black-box AI models have different APIs, and so do different attack algorithms. On top of that, I have some ideas how to tweak/change current attacks to better suit the APIs of different MLaaS threat models. AdvPipe is intended to be a modular pipeline, that would incorporate various attack regimes, target models and attack algorithms into single framework.

More on this [here](adv-pipe.md)


## Setup
- Tested on cuda 11.3, python 3.8
```
# Install poetry (you can also try pip-installing it, but this is the official way)
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# install all dependencies from lock-file
$ poetry install
```
That's all, really :) 

## Running experiment
Create experiment YAML configuration and pass it to advpipe_attack.py
```
$ cd src/advpipe
$ python advpipe_attack.py --config=attack_config/square_attack_resnet18.yaml
```
 


## Attacks

To get a sense of what this complicated title means in practice, checkout some of my other repos with [PoC](https://en.wikipedia.org/wiki/Proof_of_concept) adversarial attacks on Google Vision API:
- [Square attack](https://github.com/kubic71/square-attack)
- [TREMBA](https://github.com/kubic71/TREMBA)
- [RayS](https://github.com/kubic71/RayS)
- [Sparse-rs](https://github.com/kubic71/sparse-rs)


