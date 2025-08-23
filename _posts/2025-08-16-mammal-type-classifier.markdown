---
layout: post
comments: false
title: "Mammal Type Classifier"
excerpt: "How does a machine learn what connects a bear to a leopard—but not a zebra?"
date:   2025-08-16 00:00:00
mathjax: false
tags: 
  - classification
  - nature
  - neural network
---

How does a machine learn what connects a bear to a leopard—but not a zebra?

That question sparked the motivation for this project: I wanted to see if a model could capture the general features of a group rather than just memorizing individual animals. To explore this, I built a mammal type classifier designed to recognize shared traits across species. The goal was to train a model that, when shown a new image, could correctly place it into the right group.

The challenge is clear: bears and leopards look very different from each other, yet they both belong to Carnivores, while a zebra sits firmly with Ungulates. To give a visual sense of what I’m aiming for, I created the 3D Mammal Space below. Each point represents an animal, and the closer two points are, the more alike the model considers them. The clusters make it easy to distinguish groups—this structured separation is the kind of pattern I hope the model will uncover on its own after training.

<iframe src="/assets/3D_mammal_space.html" width="100%" height="600" style="border:none;"></iframe>

### Data 
> To build my dataset, I pulled images from DuckDuckGo using a custom API, aiming for about 100 images per animal. Not every download worked (some files were corrupted), so I added error handling to skip them, which left me with closer to 85–100 images per animal. That turned out to be enough for the experiment I had in mind. Once collected, I resized all images to 192×192 pixels to keep them consistent and manageable, then converted them into PyTorch tensors for efficient training. Finally, I organized everything into a clean directory structure with an 80/20 train–validation split, making it simple to feed directly into the model later

<p align="center">
  <img src="/assets/output.png" alt="Mammal Image Grid" width="600" height="450" />
</p>

### Algorithm
> For the modeling stage, I compared three leading pre-trained Convolutional Neural Networks: ResNet, EfficientNet, and MobileNet. ResNet provided a strong baseline with its ability to train deep networks. EfficientNet focused on balancing accuracy and efficiency through careful scaling. MobileNet was designed for speed and lightweight use, making it ideal for testing compact models. Evaluating all three on the same dataset allowed me to directly compare their strengths, observe the trade-offs between speed and accuracy, and assess how architectural choices influence classification results. IThis process also gave me hands-on experience with PyTorch and Fastai, two widely used frameworks in machine learning.

### Compute
> I trained all three models on my local machine using only CPU resources (AMD Ryzen 7 7730U with 16 GB of RAM). While this setup was sufficient for the experiment, it came at the cost of longer training times: MobileNet v3 Large finished in about 12 minutes, ResNet-18 in 18 minutes, and EfficientNet-B0 took the longest at 23 minutes. These results highlighted the efficiency differences between architectures, with MobileNet’s lightweight design showing clear speed advantages. To keep the comparison fair, I standardized the setup across all models: the same dataset of five mammal types, images resized to 192×192 pixels, and consistent training parameters—15 epochs, a learning rate of 0.001, the Adam optimizer, cross-entropy loss, and a StepLR scheduler. This consistency ensured that any performance differences could be traced back to the models themselves rather than variations in setup. In future iterations, I plan to experiment with GPU acceleration, which would dramatically reduce training time and open the door for deeper exploration.

