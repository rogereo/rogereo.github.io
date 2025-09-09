---
layout: post
comments: false
title: "Mammal Type Classifier"
excerpt: "How does a machine learn that a leopard has more in common with a wolf than with a zebra?"
date:   2025-09-09 00:00:00
mathjax: false
tags: 
  - classification
  - nature
  - neural network
---

### How does a machine learn that a leopard shares more in common with a wolf than with a zebra?

> This project set out to explore whether a model could move beyond memorizing individual animals and instead capture the shared features that define mammal groups. In biology, leopards and wolves are classified as carnivores while zebras are ungulates, distinctions grounded in traits like teeth, jaws, limbs, and skull structure. Without anatomical data available, this becomes a question of whether a computer vision model, trained only on images, can learn the underlying biological groupings just from visual patterns. To test this, I built a mammal type classifier using supervised learning, aiming to train it to recognize subtle similarities across species and correctly classify new ones. To illustrate the goal, I created a 3D Mammal Space, where each animal is represented as a point. In this space, proximity reflects similarity, and clusters emerge for each group—an intuitive visualization of the structured separation I hope the model will learn to discover on its own.

<div class="responsive-embed">
  <iframe src="/assets/3D_mammal_space.html" width="100%" height="500" style="border:none;"></iframe>
</div>

### Data 
> To build my dataset, I pulled images from DuckDuckGo using a custom API, aiming for about 100 images per animal. Not every download worked (some files were corrupted), so I added error handling to skip them, which left me with closer to 85–100 images per animal. That turned out to be enough for the experiment I had in mind. Once collected, I resized all images to 192×192 pixels to keep them consistent and manageable, then converted them into PyTorch tensors for efficient training. Finally, I organized everything into a clean directory structure with an 80/20 train–validation split, making it simple to feed directly into the model later

<p align="center">
  <img src="/assets/MammalImageGrid.png" alt="Mammal Image Grid" loading="lazy" decoding="async">
</p>

### Algorithm
> For the modeling stage, I compared three leading pre-trained Convolutional Neural Networks: ResNet, EfficientNet, and MobileNet. ResNet provided a strong baseline with its ability to train deep networks. EfficientNet focused on balancing accuracy and efficiency through careful scaling. MobileNet was designed for speed and lightweight use, making it ideal for testing compact models. Evaluating all three on the same dataset allowed me to directly compare their strengths, observe the trade-offs between speed and accuracy, and assess how architectural choices influence classification results. This process also gave me hands-on experience with PyTorch and Fastai, two widely used frameworks in machine learning.

### Compute
> I trained all three models on my local machine using only CPU resources (AMD Ryzen 7 7730U with 16 GB of RAM). While this setup was sufficient for the experiment, it came at the cost of longer training times: MobileNet v3 Large finished in about 12 minutes, ResNet-18 in 18 minutes, and EfficientNet-B0 took the longest at 23 minutes. These results highlighted the efficiency differences between architectures, with MobileNet’s lightweight design showing clear speed advantages. To keep the comparison fair, I standardized the setup across all models: the same dataset of five mammal types, images resized to 192×192 pixels, and consistent training parameters—15 epochs, a learning rate of 0.001, the Adam optimizer, cross-entropy loss, and a StepLR scheduler. This consistency ensured that any performance differences could be traced back to the models themselves rather than variations in setup. In future iterations, I plan to experiment with GPU acceleration, which would dramatically reduce training time and open the door for deeper exploration.

### Evaluation

##### **Project Structure**
> The project follows a simple and modular structure with two main directories: app and model. The app directory stores the trained models and contains the Gradio application used for deployment. The model directory holds the dataset (organized into training and validation sets) along with a notebook for training and evaluating the models. The full directory structure is shown below and can also be viewed on [Github](https://github.com/rogereo/Mammal-Type-Classifier.git)


<div style="
    padding: 10px;
    border-radius: 8px;
    width: 300px;
    background-color: transparent;
">
<pre style="
    background-color: #0d1117;   /* dark background */
    color: #e6edf3;              /* off-white text */
    font-family: helvetica;
    font-size: 14px;
    padding: 12px;
    border-radius: 6px;
">
Mammal-Type-Classifier/
├── app/
│   ├── models/ 
│   ├── app.py  
│   └── requirements.txt 
├── model/
│   ├── dataset/
│   │   ├── train/  
│   │   └── val/  
│   ├── mammaltypeclassifier.ipynb  
│   └── utils/
└── README.md
</pre>
</div>

##### **Validation**

> To evaluate the models, I used a validation dataset of 464 images, representing 20% of the original split. 
Each trained model was run once across this set, generating predictions that I compared directly with the 
true labels. From this, I calculated accuracy and average loss, giving me a first look at how well each architecture 
performed. My goal at this stage was not exhaustive experimentation but rather to complete the end-to-end process of 
training and evaluation as quickly as possible. Because training was limited to my local CPU, I only performed a 
single evaluation pass per model. For future iterations, I plan to leverage GPU resources, which will allow me to 
run more experiments, tune hyperparameters more freely, and explore advanced validation methods.


<figure class="responsive">
  <div>
    <figcaption><b>ResNet</b></figcaption>
    <p>----------</p>
    <table style="border: none; font-family: helvetica;">
      <tr><td>Total Images:</td><td style="text-align: right;">464</td></tr>
      <tr><td>Correct Prediction:</td><td style="text-align: right;">417</td></tr>
      <tr><td>Accuracy:</td><td style="text-align: right;">89.9%</td></tr>
      <tr><td>Loss:</td><td style="text-align: right;">0.39</td></tr>
    </table>
  </div>
  <img src="/assets/resnet_confusionmatrix.png"
       alt="ResNet confusion matrix"
       title="ResNet"
       loading="lazy"
       decoding="async">
</figure>

<figure class="responsive">
  <div>
    <figcaption><b>EfficientNet</b></figcaption>
    <p>----------</p>
    <table style="border: none; font-family: helvetica;">
      <tr><td>Total Images:</td><td style="text-align: right;">464</td></tr>
      <tr><td>Correct Prediction:</td><td style="text-align: right;">432</td></tr>
      <tr><td>Accuracy:</td><td style="text-align: right;">93.1%</td></tr>
      <tr><td>Loss:</td><td style="text-align: right;">0.22</td></tr>
    </table>
  </div>
  <img src="/assets/efficientnet_confusionmatrix.png"
       alt="EfficientNet confusion matrix"
       title="EfficientNet"
       loading="lazy"
       decoding="async">
</figure>

<figure class="responsive">
  <div>
    <figcaption><b>MobileNet</b></figcaption>
    <p>----------</p>
    <table style="border: none; font-family: helvetica;">
      <tr><td>Total Images:</td><td style="text-align: right;">464</td></tr>
      <tr><td>Correct Prediction:</td><td style="text-align: right;">422</td></tr>
      <tr><td>Accuracy:</td><td style="text-align: right;">90.8%</td></tr>
      <tr><td>Loss:</td><td style="text-align: right;">0.33</td></tr>
    </table>
  </div>
  <img src="/assets/mobilenet_confusionmatrix.png"
       alt="MobileNet confusion matrix"
       title="MobileNet"
       loading="lazy"
       decoding="async">
</figure>

### Deployment
> After training and evaluation, I saved the models as pickled files so they could be reused without retraining and integrated them into a Gradio web app. The application’s core feature is a side-by-side comparison, letting users test an image across ResNet, EfficientNet, and MobileNet while seeing the predicted label, actual label, cross-entropy loss, and prediction probability for each. To make the experience more engaging, I added Gemini LLM features: an automated performance summary that highlights which models were correct and how they differed, and a fun fact about the animal whenever a model gets the prediction right. Together, these features turn a simple classifier into an interactive, educational tool that not only demonstrates model performance but also makes the results more insightful and rewarding to explore.

Leopard classification
<p align="center">
<img src="/assets/app_leopard_.png" width="100%" height="430" alt="Carnivore Leopard" title="Leopard" />
</p>

Wolf classification
<p align="center">
<img src="/assets/app_wolf.png" width="100%"height="430"
alt="Carnivore Wolf" title="Wolf" />
</p>

Zebra classification
<p align="center">
<img src="/assets/app_zebra_.png" width="100%" height="430" alt="Ungulate Zebra" title="Zebra" />
</p>

### Conclusion
>In the end, the project gave me a partial but meaningful answer to my original question. By building a CNN-based mammal classifier, I was able to demonstrate how a model can learn to group animals like leopards and wolves together while distinguishing them from zebras. What I don’t yet have is a full grasp of the mathematics behind how these connections are formed, but I now have a working system that makes the process visible through application. The real achievement lies in transforming an abstract question into something tangible: an end-to-end classifier and interactive app that not only validates the idea in practice but also lays the foundation for deeper exploration.