# [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](semanticscholar.org/paper/Adversarial-Attacks-and-Defenses-in-Images%2C-Graphs-Xu-Ma/6ad5f1d88534715051c6aba7436d60bdf65337e8)
- 2020, 108 citations

### Vulnerable networks
- CNN
- GCN (graph convolutional networks) - used in fraud detection
    - only necessary to change couple of edges
- 

### Counter-measures
- Gradient masking
- Robust optimization
- Adversary detection


Deep neural-nets reason differntly -> understanding adversarial attacks should help understand this difference

## Threat model
### Adversary's goal
- Poisoning attack - change the behavior of DNN by modifying/inserting few train examples
    - public honeypot - collection of training data for malware detectors
- evasion attack - craft fake examples classifier cannot recognize
    - targeted 
    - untargeted

### Adversary's knowledge
- White-box attack - widely studied, easily analyzed mathematically
- Black-box attack - practical
- Semi-white (gray) box attack - train generative model in white-box setting, then use in black-box scenario (TREMBA)

### Victim models
- conventional machine learning models - SVM, Naive-Bayes
- DNN - not well understood how they work, studying security necessary
    - FC
    - CNN - sparse version of FC
    - GCN
    - RNN


## Security evaluation
### Robustness
- **Minimal pertubation** - The smallest pertubation to fool the network
    - $\delta_{min} = \underset{\delta}{\argmin} \lVert \delta \rVert$ s.t. $F(x + \delta) \neq y$

- **Robustness** - The norm of minimal pertubation on particular example
    - $r(x, F) = \lVert \delta_{min} \rVert$
