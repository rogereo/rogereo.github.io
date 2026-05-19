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

<p align="center">
  <img src="/assets/tracer-chart.png" loading="lazy" decoding="async">
</p>

> Earlier this year, Andrej Karpathy released a small repo called AutoResearch: a pattern where you point an LLM at a research codebase and let it propose, run, and evaluate experiments on its own, in a loop. Most writeups that followed asked the same question: can it beat the benchmarks?
I wanted to ask a different one. I came across the repo and wanted to reimplement it, partly to understand it, but mostly because I was curious about something else. An ML problem is a fixed environment. The goal never changes: improve the score. So what does a small LLM actually do inside that box? What does it try first, before any feedback exists? Where does it get stuck? When does it surprise you? What does it never think to try?
To find out, I built a minimal version of the pattern, pointed it at the Spaceship Titanic Kaggle problem, gave it a cheap model, and read the trace.
154 iterations and $2.34 later, here I am.

### Data 

<p align="center">
  <img src="/assets/kaggle.png" loading="lazy" decoding="async">
</p>

> [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) is a sci-fi remix of the original Kaggle Titanic challenge. A spaceship carrying thousands of passengers hit a spacetime anomaly. Instead of predicting survival after a shipwreck, the goal is to predict whether each passenger was transported to another dimension after an interstellar accident.
Binary classification. 8,693 training rows. 14 columns covering passenger identity, origin and destination, travel status, cabin info, demographics, and spending behaviour.
I picked this over the original Titanic for two reasons.
First, the features hide useful structure. PassengerId encodes travel groups. Cabin splits into deck, room, and side. The spending columns interact strongly with CryoSleep, because passengers that are 'sleeping' cannot spend. None of this is handed to you in the column descriptions. An agent has to discover it.
Second, I like the idea of space exploration. 


### Algorithm
> The setup has two layers. An outer loop runs the research process. An inner loop is where the agent picks the actual ML approach.

##### Outer Loop

> The outer loop has four parts: `prepare.py` is the immutable evaluator that creates the 80/20 split and exposes `evaluate(predict_fn)`. `program.md` is the brief the agent reads every turn. `workspace/train.py` is the sandbox, the only file the agent rewrites. `orchestrator.py` is the harness that assembles the prompt, calls Claude Haiku 4.5, writes the new `train.py`, and runs it with a 60-second timeout. The new score is compared to the best so far. If it matches or beats it, the change is committed. If it crashes or regresses, Git resets the file. `train.py` can only move forward. The run stops after 15 iterations, $10 of cost, or 5 turns without improvement.

<p align="center">
  <img src="/assets/outer-loop.png" alt="The Loop: prepare, program, train, orchestrate" loading="lazy" decoding="async">
</p>

##### Agent Reasoning
> Each model response is parsed into six fields: reflection on the previous result, observations about what the agent notices, a hypothesis the next experiment will test, a plan for the specific change, the full new `train.py`, and any notes worth carrying forward. Every iteration takes the shape of a small research cycle: observe, hypothesise, act, evaluate.

<p align="center">
  <img src="/assets/agent-reasoning.png" alt="Agent reasoning: reflection, observations, hypothesis, plan, code, notes append" loading="lazy" decoding="async">
</p>

##### Inner Loop

> The inner loop is the model building strategy inside `train.py`. That part isn't fixed. The agent decides whether to do EDA or jump straight to modelling, whether to reach for logistic regression, random forests, XGBoost, feature engineering, imputation, ensembling, or hyperparameter tuning.

### Compute

> I ran the agent on Claude Haiku 4.5 through the Anthropic API. The cheapest tier Anthropic offers, weaker than Sonnet or Opus at reasoning. If Haiku is enough for tabular ML on a small dataset, that's a useful result. If it isn't, that tells you where autoresearch breaks. The sandbox runs locally on my CPU. Each `train.py` execution gets a 60-second timeout. Anything longer is killed and reverted. The budget pressure forces the agent to keep its experiments small, which keeps the trace readable. Across 24 runs and 154 iterations, total cost: $2.34

### Evaluation

> 154 iterations is enough to read the run as a sequence. The agent moved the score from 0.7809 to 0.8235: 63 kept, 73 reverted, 6 crashes, 12 parser failures. Three phases: a slow climb through feature engineering, one sharp jump from logistic regression to XGBoost, then a long flat tail.

<p align="center">
  <img src="/assets/tracer-chart.png" loading="lazy" decoding="async">
</p>

##### Breakthrough

> For 20 iterations the agent stuck with logistic regression. Then in iteration 21 it changed strategy and the score jumped from 0.7907 to 0.8143. The biggest single move of the run. What's interesting isn't that it reached for XGBoost. It's that it named the reason first: the linear model couldn't capture the structure in the data

<p align="center">
  <img src="/assets/iteration-21.png" loading="lazy" decoding="async">
</p>

##### Where it got stuck

> By iteration 76 the agent hit 0.8235 and stayed there for the next 78 iterations. It recognized the plateau correctly but kept proposing feature variants instead of another structural pivot like the XGBoost move

<p align="center">
  <img src="/assets/iteration-51.png" loading="lazy" decoding="async">
</p>

### Deployment
> The full code is on [GitHub](https://github.com/rogereo/spaceship-titanic-autoresearch). Clone the repo and follow the setup instructions. A full run takes 30 to 60 minutes and costs a few cents. I found the trace most interesting. Every iteration writes one JSONL line with the agent's reflection, hypothesis, plan, code, stdout, and final status. An interactive viewer in the repo lets you click any iteration to see what it was thinking. If you don't want to run the loop yourself, read mine.

### Conclusion
> Give a small LLM control of its own ML experiments and it runs them. It improves the score. It plateaus, notices the plateau, and keeps running variations of what already worked. Two things I'd try next time. Run it on a stronger model (Sonnet, or Opus) to see whether the plateaus are a Haiku problem or a framework problem. Point the same loop at a harder problem, something with images or text, where feature engineering won't be enough. 154 experiments. $2.34. Cheap enough to be worth doing again.

-----
References
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) - Andrej Karpathy's original autoresearch repo
- [Guide to Autoresearch](https://www.datacamp.com/tutorial/guide-to-autoresearch) - DataCamp tutorial I read to understand the pattern
- [Karpathy's announcement thread](https://x.com/karpathy/status/2030371219518931079) - the original X post introducing the repo
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) - Kaggle competition page
- [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) - the model used as the agent
- [Anthropic API docs](https://docs.claude.com) - for setting up the API key and rate-limiting calls
- [XGBoost docs](https://xgboost.readthedocs.io/) - the model class the agent landed on
- [scikit-learn](https://scikit-learn.org/) - the library the sandbox runs on