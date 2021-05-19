# Ideas about adversarial attack on Cloud-based MLaaS

## Attack pipeline
Many of the ideas bellow can be in principle combined. This may improve attack efficiency and success rate.
Create versatile modular pipeline, that would stack different attack algorithms for better attack performance.

## Metacognitive confidence of neural-nets 
- make a neural net also output it's confidence of classification

### Captcha
- captcha recognizer that can reliably assess the difficutly of one captcha instance
- hard captcha could be skipped automatically by the system

### Adversarial attacks 
- given a set of original examples, choose the one that will most likely result in a successful attack
- could potentially safe valuable queries
    - attack less likely to be detected 
- could this be defended by adversarial example detector?



## Map GVision outputted image captions to ImageNet labels
The idea is to map the problem of attacking a cloud-based black-box classifier with tens of thousands of possible labels to 
### Map 


## Model-stealing to train a substitute model
Use model-stealing techniques to efficiently train a subtitute model. Transfer adversarial examples crafted on the obtained subsitute.

### Model-stealing
- [CloudLeak](https://www.semanticscholar.org/paper/CloudLeak%3A-Large-Scale-Deep-Learning-Models-Through-Yu-Yang/4d548fd21aad60e3052455e22b7a57cc1f06e3c3)
    - Authors test their algorithm on Microsoft, Face++, IBM, Google and Clarifai MLaaS platforms

### 



## Multiple-examples adversarial attack
### Problem statement
Given a set of original examples, produce one successful adversarial example with the fewest number of queries to the target that is being attacked.

The hypothesis is, that when we are given multiple possible starting points, we have another degree of freedom in the adversarial attack search space, which should make the task of finding one adversarial example easier.


### Pick the most promising starting point
If we for example want to find a single adversarial image of a cat, we can take diverse 100 images of different cats and decide, which one is the most vulnerable to adversarial pertubation and execute a specific attack only on this single most promising cat image.

The question of course is, how best to select it. Maybe train a classifier on data from simulated local attacks?


### Parallel-scheduling iterative approach
In the case of iterative attacks, we can add on top another meta-algorithm, that will execute one step of optimization one image from the set. It will observe the black-box output, that will guide the following decisions of allocating the remaining budget of queries.

This can be framed as a bandit RL problem.

If the attack algorithm is stochastic and if the initial set contains copies of the original image, we would be effectively searching the pertubation space in parallel, prioritizing the more successful random initializations.

Some genetic-style algorithm could be also used on the population of running instances. I will have to think  more about the crossover operator.


### Multiple algorithms
We can add another degree of freedom by having multiple toolset of adversarial attack algorithms. This is 2D bandit.


### Transferable set-Universal pertubation
In the case of transfer-attacks, We can maybe have a better chance of finding transferable adversarial pertubation by optimizing the sum loss on all images in the set, to which the same pertubation will be applied. 




## Initialize iterative attack by the output of transfer-based attack algorithm 

