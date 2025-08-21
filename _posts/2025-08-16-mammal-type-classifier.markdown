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

## Data 
> To build my dataset, I pulled images from DuckDuckGo using a custom API, aiming for about 100 images per animal. Not every download worked (some files were corrupted), so I added error handling to skip them, which left me with closer to 85–100 images per animal. That turned out to be enough for the experiment I had in mind. Once collected, I resized all images to 192×192 pixels to keep them consistent and manageable, then converted them into PyTorch tensors for efficient training. Finally, I organized everything into a clean directory structure with an 80/20 train–validation split, making it simple to feed directly into the model later

<img src="/assets/output.png" alt="Mammal Image Grid" width="800" height="650" />

## Algorithm
> For the modeling stage, I decided to compare three different pre-trained Convolutional Neural Networks (ResNet, EfficientNet, and MobileNet) each chosen because they represent a distinct philosophy in deep learning design. ResNet served as a solid baseline, with its proven ability to learn deep features through residual connections. EfficientNet offered a more modern approach, balancing accuracy and efficiency by carefully scaling its depth, width, and resolution. MobileNet, on the other hand, was built for speed and lightweight deployment, making it ideal for testing how well a compact model could handle this challenge. Running all three on the same dataset gave me the chance to directly compare their strengths, explore the 
trade-offs between speed and accuracy, and see how architectural choices affect classification results. It also allowed me to experiment hands-on with PyTorch and Fastai, adding practical experience with two of the most widely used frameworks in machine learning.
