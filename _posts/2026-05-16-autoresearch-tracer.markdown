---
layout: post
comments: false
title: "One Agent, 154 Experiments"
excerpt: "What happens when you give a small LLM control of its own ML experiments?"
date:   2026-05-16 00:00:00
mathjax: false
tags:
  - agents
  - llm
  - machine learning
---

### Question
##### What happens when you give a small LLM control of its own ML experiments?
> Earlier this year, Andrej Karpathy released a small repo called AutoResearch: a pattern where you point an LLM at a research codebase and let it propose, run, and evaluate experiments on its own, in a loop. Most writeups that followed asked the same question: can it beat the benchmarks?
I wanted to ask a different one. I came across the repo and wanted to reimplement it, partly to understand it, but mostly because I was curious about something else. An ML problem is a fixed environment. The goal never changes: improve the score. So what does a small LLM actually do inside that box? What does it try first, before any feedback exists? Where does it get stuck? When does it surprise you? What does it never think to try?
To find out, I built a minimal version of the pattern, pointed it at the Spaceship Titanic Kaggle problem, gave it a cheap model, and read the trace.
154 iterations and $2.34 later, here I am.

<p align="center">
  <img src="/assets/trace-viewer.png" alt="NASA" loading="lazy" decoding="async">
</p>

### Data 
> [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) is a sci-fi remix of the original Kaggle Titanic challenge. A spaceship carrying thousands of passengers hit a spacetime anomaly. Instead of predicting survival after a shipwreck, the goal is to predict whether each passenger was transported to another dimension after an interstellar accident.
Binary classification. 8,693 training rows. 14 columns covering passenger identity, origin and destination, travel status, cabin info, demographics, and spending behaviour.
I picked this over the original Titanic for two reasons.
First, the features hide useful structure. PassengerId encodes travel groups. Cabin splits into deck, room, and side. The spending columns interact strongly with CryoSleep, because passengers that are 'sleeping' cannot spend. None of this is handed to you in the column descriptions. An agent has to discover it.
Second, I like the idea of space exploration. 
The dataset never moves during the experiment. A fixed train/validation split (80/20, stratified, random_state=42) lives in prepare.py, an immutable file the agent can't touch. Every score the agent produces is comparable to every other. That's what makes the trace readable as a sequence.


### Algorithm
The setup has two layers. An outer loop runs the research process. An inner loop is where the agent picks the actual ML approach.

#### The outer loop: the autoresearch harness

The outer loop lets the agent run experiments independently. It has four parts:

- `prepare.py` is the evaluator. It creates a fixed 80/20 train-validation split with `random_state=42` and exposes one function, `evaluate(predict_fn)`. The file never changes, so every score is directly comparable.
- `program.md` is the brief. It's the human written instruction file the agent reads every turn, defining the problem, rules, allowed libraries, timeout, required output format, and research directives.
- `workspace/train.py` is the sandbox. The only file the agent rewrites. Its contract is simple: call `prepare.evaluate(predict_fn)` once and print `VAL_ACCURACY: 0.XXXX` as the final line.
- `orchestrator.py` is the harness. Each turn it reads the brief, current code, notes, and recent history, sends them to Claude Haiku 4.5, parses the response, writes the new `train.py`, and runs it with a 60-second timeout.

The new score is then compared to the best score so far. If it matches or beats it, the change is committed. If it crashes or regresses, Git resets the file. Failed experiments disappear. `train.py` can only move forward.

The run stops after 15 iterations, $10 of cost, or 5 turns without improvement.

#### The agent's reasoning format

Each model response is parsed into six fields:

- **Reflection.** What the previous result showed.
- **Observations.** What the agent notices.
- **Hypothesis.** The claim the next experiment will test.
- **Plan.** The specific change it will make.
- **Code.** The full new `train.py`.
- **Notes append.** Anything worth carrying forward.

Every iteration takes the shape of a small research cycle: observe, hypothesise, act, evaluate.

#### The inner loop: whatever the agent discovers

The inner loop is the model building strategy inside `train.py`. That part isn't fixed. The agent decides whether to do EDA or jump straight to modelling, whether to reach for logistic regression, random forests, XGBoost, feature engineering, imputation, ensembling, or hyperparameter tuning.

My job wasn't to pick the algorithm in advance. My job was to build the loop, let the agent search, and report what it found.

### Compute
> 

### Evaluation
> e 

### Deployment
> 

### Conclusion
> 

-----
References
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) - Andrej Karpathy's original autoresearch repo
- [A Guide to Andrej Karpathy's AutoResearch](https://www.datacamp.com/tutorial/guide-to-autoresearch) - DataCamp tutorial I used as reference to understand the pattern
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) - Kaggle competition page
- [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) - the model used as the agent