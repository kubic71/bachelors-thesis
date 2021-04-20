# Introduction
## The rise artificial neural networks
In recent years there have been enourmous surge in applications of artificial intelligence technologies based on neural networks to various fields. One of the biggest milestones that kick-started today's AI revolution have undoubtedly been the year 2012, when AlexNet won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with its CNN architecture, which have been previously dominated by hand-crafted specialized image-labeling systems.

Artificial neural networks, which are inspired by their biological analog, had been been known and researched for a long time. Even though they enjoyed a lot of enthusiasm in the beginning, they have been neglected as unpromising direction towards general intelligence. More classical approaches like SVM and rule-based AI systems showed better performance and computational efficiency on AI benchmarks of the time. 

What changed the game has been the available computational power that came with the Moore's law and the usage of GPUs, mainly their parallel nature in accelerating matrix multiplication operations heavily used in neural networks.

Another factor that helped the rise of neural networks has been the availability of large datasets like ImageNet, which contains 1,281,167 images for training and 50,000 images for validation organised in 1,000 categories. 

Availibility of large amounts of labeled and unlabeled data, sometimes refered to as "Big data", only seems to get better. Large part of our lifes has moved to the virtual space thanks to the internet. Businesses had started to realize the value of the enormous amount of traffic generated every day. 

What have been previously limited to academic cirles have quickly become mainstream. Artificial neural networks have proven to be very versatile and have quickly been successfuly applied to wide range of problems. New neural network architectures and new training regimes allowed training deeper networks, which gave rise to a new field of machine learning called Deep learning. 

Deep neural networks have shown state-of-the-art performance in machine translation, human voice synthesis and recognition, drug composition, particle accelerator data analysis, recommender systems, algorithmic trading, reinforcement learning and many other areas.

## Concerns
Large-scale deployment of neural network systems has been critized by some for their inherent unexplainability. It is often hard to pinpoint why neural network behaves in some way and why it makes certain decisions. One problem is the role of training data, where possible biases may be negatively and unexpectably reflected in the behavior of the AI system. Another problem is the performance on out-of-distribution examples, where network inference takes place on different kind of data than used in training stage.

Those concerns lead people to the study the robustness of AI systems. It turned out that image recognition CNN networks are especially susceptible to the so-called adversarial attacks, where the input is pertubed imperceptibly, but the output of the network changes wildly. Similar kinds of attacks have since been demontrated in other areas like speech recognition, natural language processing or in reinforcement learning.

## Adversarial attacks
This has lead to cat-and-mouse game of various attack strategies and following defenses proposed to mitigate them.

Neural networks can be attacked at different stages:
- training
- testing
- deployment

Training attacks exploit training dataset manipulation, sometimes called dataset poisoning, to change the behavior of the resulting trained network.

Testing attacks don't modify trained neural network, but often use the knowledge of the underlying architecture to craft adversarial examples which fool the system. 

Deployment attacks deal with real black-box AI systems, where the exact details of the system are usually hidden from attacker. Nevertheless partly because similar neural network architectures are used in the same classes of problems, some vulnerabilities in testing scenario can still be exploited in deployment, even though the exact network, architecture and its output is hidden from the attacker.

The purpose of this thesis is to explore the applicability of certain classes of testing attacks on real-world deployed AI systems. For simplicity many kinds of SOTA adversarial attacks have only been been explored in the testing regime, but haven't been applied to truly black-box systems. 

The main aim of the thesis will be to test different types types of testing attacks on AI Saas providers like Google Vision API, Amazon rekognition or Clarify. Understandably, attacking unknown system will be harder than attacking known neural network in the testing stage. We will try to measure this attack difficutly increase. This information could be used for selecting the most promising attack to a specific situation. 

Many SOTA testing attacks are not designed to attack specific deployed systems, so some attacks will need to be slightly changed to be used. We will explore different ways to modify existing attacks and evaluate them.

If some of those services are proven to be vulnerable, this would in fact have a huge impact on all downstream applications using those Saas APIs. Content moderation mechanisms, which rely in a large part on automatic detection, could be circumvented.

