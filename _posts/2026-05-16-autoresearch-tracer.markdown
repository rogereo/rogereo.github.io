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
  <img src="/assets/tracer-ui.png" loading="lazy" decoding="async">
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
The dataset never moves during the experiment. A fixed train/validation split (80/20, stratified, random_state=42) lives in prepare.py, an immutable file the agent can't touch. Every score the agent produces is comparable to every other. That's what makes the trace readable as a sequence.


### Algorithm
> The setup has two layers. An outer loop runs the research process. An inner loop is where the agent picks the actual ML approach.

#### The outer loop: the autoresearch harness

> The outer loop lets the agent run experiments independently. It has four parts:

- `prepare.py` is the evaluator. It creates a fixed 80/20 train-validation split with `random_state=42` and exposes one function, `evaluate(predict_fn)`. The file never changes, so every score is directly comparable.
- `program.md` is the brief. It's the human written instruction file the agent reads every turn, defining the problem, rules, allowed libraries, timeout, required output format, and research directives.
- `workspace/train.py` is the sandbox. The only file the agent rewrites. Its contract is simple: call `prepare.evaluate(predict_fn)` once and print `VAL_ACCURACY: 0.XXXX` as the final line.
- `orchestrator.py` is the harness. Each turn it reads the brief, current code, notes, and recent history, sends them to Claude Haiku 4.5, parses the response, writes the new `train.py`, and runs it with a 60-second timeout.

> The new score is then compared to the best score so far. If it matches or beats it, the change is committed. If it crashes or regresses, Git resets the file. Failed experiments disappear. `train.py` can only move forward.

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

> The inner loop is the model building strategy inside `train.py`. That part isn't fixed. The agent decides whether to do EDA or jump straight to modelling, whether to reach for logistic regression, random forests, XGBoost, feature engineering, imputation, ensembling, or hyperparameter tuning.

### Compute

> I ran the agent on Claude Haiku 4.5 through the Anthropic API. The cheap, fast tier, nominally weaker than Sonnet at reasoning. If autoresearch only works with the strongest model, that's a useful boundary. If Haiku is enough for tabular ML on a small dataset, that's also useful, and probably more relevant to most builders. The sandbox runs locally on my CPU. No GPU. Each `train.py` execution gets a 60-second timeout. Anything longer is killed and reverted. The budget pressure forces the agent to keep its experiments small, which keeps the trace readable. Across 24 runs and 154 iterations, total cost: $2.34

### Evaluation

> 154 iterations is enough to read the run as a sequence. The agent moved the score from 0.7809 to 0.8235: 63 kept, 73 reverted, 6 crashes, 12 parser failures. Top public Kaggle scores sit around 0.81 to 0.82, so it landed where a decent first time entrant would. Three phases: a slow climb through feature engineering, one sharp jump from logistic regression to XGBoost, then a long flat tail. 

<p align="center">
  <img src="/assets/tracer-chart.png" loading="lazy" decoding="async">
</p>

#### The breakthrough

> For 20 iterations the agent stuck with logistic regression. Then in iteration 21 it changed strategy and the score jumped from 0.7907 to 0.8143. The biggest single move of the run. What's interesting isn't that it reached for XGBoost. It's that it named the reason first: the linear model couldn't capture the structure in the data

<p align="center">
  <img src="/assets/iteration-21.png" loading="lazy" decoding="async">
</p>

#### Where it got stuck

> By iteration 76 the agent hit 0.8235 and stayed there for the next 78 iterations. It recognized the plateau correctly but kept proposing feature variants instead of another structural pivot like the XGBoost move

<p align="center">
  <img src="/assets/iteration-51.png" loading="lazy" decoding="async">
</p>


### Deployment
> 

### Conclusion
> One pattern across the run: when the agent won, it usually knew why. Eight of the nine times it beat the running best, it had named the reason before running the change. The same confidence showed up on plenty of reverts though, so the read-ahead worked on wins but not losses.

-----
References
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) - Andrej Karpathy's original autoresearch repo
- [Guide to Autoresearch](https://www.datacamp.com/tutorial/guide-to-autoresearch) - DataCamp tutorial I read to understand the pattern
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) - Kaggle competition page
- [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) - the model used as the agent