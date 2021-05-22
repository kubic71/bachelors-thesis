# AdvPive
- [OrgPad diagram](https://orgpad.com/s/HjtFcaxE4AM)

There are many different approaches to adversarial attacks that can be combined together, hopefully to be more powerful.

I also have many ideas of possible ways to change or tweak current attacks, how to make them more versatile, potent etc.

Therefore a good idea would be to create modular adversarial attack pipeline, where for example one paper or attack algorithm could roughly correspond to one module.

Various attacked victim platforms (GVision, Amazon rekognition, Facebook content moderation AI) would be abstracted by some common module API. So adding a new cloud target into the pipeline would be very easy, just wrap it in the module API.

My ideas or different approaches from various papers could be incorporated into this pipeline as other separate modules or meta-layers.

I want to focus on the practicality of the attacks, so what naturally crystallized from this goal is the following **problem statement:**

*Given a set of input images (or other type of datapoints), craft a **single successful adversarial example** with the **least amount of queries** to the target black-box.*

Creating this general pipeline is obviously very ambitious goal and I can only do so much before my thesis deadline. Nevertheless, this extensible modular approach would make it very easy to add new 2021 SOTA attacks into my framework (when they come out after the summer conferences) and this framework could be very useful long after I'm finished with my thesis.
