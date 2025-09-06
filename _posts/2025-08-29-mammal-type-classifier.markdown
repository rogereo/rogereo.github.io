---
layout: post
comments: false
title: "Mammal Type Classifier"
excerpt: "How does a machine learn what connects a bear to a leopard—but not a zebra?"
date:   2025-08-29 00:00:00
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
> For the modeling stage, I compared three leading pre-trained Convolutional Neural Networks: ResNet, EfficientNet, and MobileNet. ResNet provided a strong baseline with its ability to train deep networks. EfficientNet focused on balancing accuracy and efficiency through careful scaling. MobileNet was designed for speed and lightweight use, making it ideal for testing compact models. Evaluating all three on the same dataset allowed me to directly compare their strengths, observe the trade-offs between speed and accuracy, and assess how architectural choices influence classification results. This process also gave me hands-on experience with PyTorch and Fastai, two widely used frameworks in machine learning.

### Compute
> I trained all three models on my local machine using only CPU resources (AMD Ryzen 7 7730U with 16 GB of RAM). While this setup was sufficient for the experiment, it came at the cost of longer training times: MobileNet v3 Large finished in about 12 minutes, ResNet-18 in 18 minutes, and EfficientNet-B0 took the longest at 23 minutes. These results highlighted the efficiency differences between architectures, with MobileNet’s lightweight design showing clear speed advantages. To keep the comparison fair, I standardized the setup across all models: the same dataset of five mammal types, images resized to 192×192 pixels, and consistent training parameters—15 epochs, a learning rate of 0.001, the Adam optimizer, cross-entropy loss, and a StepLR scheduler. This consistency ensured that any performance differences could be traced back to the models themselves rather than variations in setup. In future iterations, I plan to experiment with GPU acceleration, which would dramatically reduce training time and open the door for deeper exploration.

### Evaluation
> To evaluate the models, I used a validation dataset of 464 images, representing 20% of the original split. 
Each trained model was run once across this set, generating predictions that I compared directly with the 
true labels. From this, I calculated accuracy and average loss, giving me a first look at how well each architecture 
performed. My goal at this stage was not exhaustive experimentation but rather to complete the end-to-end process of 
training and evaluation as quickly as possible. Because training was limited to my local CPU, I only performed a 
single evaluation pass per model. For future iterations, I plan to leverage GPU resources, which will allow me to 
run more experiments, tune hyperparameters more freely, and explore advanced validation methods.

<div align="center">
  <figure style="display:inline-block; margin:5px;">
    <figcaption align="center"><b>ResNet</b></figcaption>
    <img src="/assets/resnet_validation.png" alt="resnet_v" width="250" height="150" title="ResNet" />
    <img src="/assets/resnet_confusionmatrix.png" alt="resnet_c" width="250" height="150" title="ResNet" />
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <figcaption align="center"><b>EfficientNet</b></figcaption>
    <img src="/assets/efficientnet_validation.png" alt="efficientnet_v" width="250" height="150"/>
    <img src="/assets/efficientnet_confusionmatrix.png" alt="efficientnet_c" width="250" height="150"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
  <figcaption align="center"><b>MobileNet</b></figcaption>
    <img src="/assets/mobilenet_validation.png" alt="mobilenet_v" width="250" height="150" />
    <img src="/assets/mobilenet_confusionmatrix.png" alt="mobilenet_c" width="250" height="150" />
  </figure>
</div>

### Deployment
> After training and evaluation, I saved the models as pickled files so they could be reused without retraining and integrated them into a Gradio web app. The application’s core feature is a side-by-side comparison, letting users test an image across ResNet, EfficientNet, and MobileNet while seeing the predicted label, actual label, cross-entropy loss, and prediction probability for each. To make the experience more engaging, I added Gemini LLM features: an automated performance summary that highlights which models were correct and how they differed, and a fun fact about the animal whenever a model gets the prediction right. Together, these features turn a simple classifier into an interactive, educational tool that not only demonstrates model performance but also makes the results more insightful and rewarding to explore.

Bear classification
<p align="center">
<img src="/assets/app_bear.png" width="800" height="500" alt="Carnivore Bear" title="Bear" />
</p>

Leopard classification
<p align="center">
<img src="/assets/app_leopard.png" width="800" height="500" alt="Carnivore Leopard" title="Leopard" />
</p>

Zebra classification
<p align="center">
<img src="/assets/app_Zebra.png" width="800" height="500" alt="Ungulate Zebra" title="Zebra" />
</p>

### Conclusion