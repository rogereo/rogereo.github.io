---
layout: post
comments: false
title: "Mammal Type Classifier"
excerpt: "How does a machine learn what connects a bear to a leopard—but not a zebra?"
date:   2025-08-09 00:00:00
mathjax: false
tags: 
  - classification
  - nature
  - neural network
---

How does a machine learn what connects a bear to a leopard—but not a zebra?

The question above captures my motivation for this project: Can a machine learn the general features of a group, rather than just memorizing individual animals? Can I build a model that can help me answer the question?

To explore this, I set out to build a mammal classifier trained to recognize shared characteristics within a group made up of unique species.

Let’s simplify the idea. Imagine two mammal groups: Carnivores and Ungulates. The model is trained with about 100 image-label pairs per animal.

Carnivores: Bears, leopards, lions, tigers, wolves
Ungulates: Deer, elk, giraffes, moose, zebras

The goal: given a new image of an animal, the model should correctly predict its group.
In the opening question, both a bear and a leopard belong to the Carnivore group, while a zebra belongs to Ungulates. The challenge is that bears and leopards look very different from each other—yet the model needs to learn what unites them, while also distinguishing them from zebras.

I created the 3D Mammal Space below to give a visual sense of what I’m aiming for with the trained model. Each point represents an animal, and the closer two points are, the more similar the model considers them. The clusters you see make it easy to distinguish the different mammal groups. Feel free to explore and connect animals to see their relationships. This visualization represents the kind of structured separation I hope the model will discover on its own after training.

<iframe src="/assets/3D_mammal_space.html" width="100%" height="600" style="border:none;"></iframe>

