# Next steps
- Things to focus on short-term
- Always ask a question: Will this be in the final thing, or are you doing unnecessary work?

# [AdvPipe](https://orgpad.com/o/CVsE8-rLlK4aqPfvPU7e1s?token=ArhyNPBrZJv7_VveqVC7U7) MVP (Minimal viable product)
- Experiment manager
    - Config loader
        - YAML
    - Dataset loader
        - find suitable dataset - maybe ImageNet
            - a lot of pretrained models are pretrained on ImageNet
        - store locally first 

- Pretrained robust ensemble
    - [L2 & Linf adversarially trained ResNet-50 with eps=8/255 ](https://github.com/MadryLab/robustness)



- Language model
    - category split:
        - organism - not-an-organism
        - organism - artifact
        - animate - inanimate
    - does organism/not-an-organism (animate/inanimate) split makes sense for ImageNet? Is the split reasonably balanced? 
        - [ImageNet class hierarchy](https://observablehq.com/@mbostock/imagenet-hierarchy)

