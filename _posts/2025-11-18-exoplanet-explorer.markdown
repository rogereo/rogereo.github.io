---
layout: post
comments: false
title: "Exoplanet Explorer"
excerpt: "If the universe speaks in light, can we learn its language well enough to find new worlds?"
date:   2025-11-18 00:00:00
mathjax: false
tags: 
  - classification
  - nature
  - space exploration
---

### Question
##### If the universe speaks in light, can we learn its language well enough to find new worlds?

<p align="center">
  <video
    autoplay
    loop
    muted
    playsinline
    preload="metadata"
    style="max-width: 960px; width: 100%; border-radius: 12px;"
  >
    <source src="/assets/Transit Method For Detecting Planets.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

> The visual above shows the [transit method](https://science.nasa.gov/mission/roman-space-telescope/transit-method/), a technique used to detect [exoplanets](https://science.nasa.gov/exoplanets/). Planets that orbit stars outside our solar system. Thousands of exoplanets have been discovered, with some estimates suggesting there could be at least one planet for every star in the galaxy. 

<p align="center">
  <img src="/assets/NasaSpaceApps.png" alt="NASA" loading="lazy" decoding="async">
</p>

> On October 4–5, I took part in the NASA Space Apps Challenge. It’s a global hackathon where participants leverage NASA’s open data to develop solutions for challenges on Earth and beyond. I stepped into the event with no experience in space data, met my teammates that morning, and jumped in relying entirely on curiosity. We teamed up to take on the [“A World Away: Hunting for Exoplanets with AI”](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/) challenge. Focused on using machine learning to identify planets orbiting distant stars. We created Exoplanet Explorer, an interactive web app that lets users explore model performance and 3D exoplanet visualizations. We’re also developing an upload feature that allows researchers to submit their own exoplanet data and run it through our models to classify planets as candidates or confirmed exoplanets. I was drawn to this challenge because it pushed my limits and connected AI to the enduring question of whether we’re alone in the universe. 

<div class="exoplanet-embed">
  <iframe
    src="https://eyes.nasa.gov/apps/exo/#/planet/Kepler-1024_b"
    title="NASA Eyes on Exoplanets Viewer"
    loading="lazy"
    allowfullscreen
    referrerpolicy="no-referrer-when-downgrade"
  ></iframe>
</div>
<p class="exoplanet-embed-note">
  Prefer a larger view? <a href="https://eyes.nasa.gov/apps/exo/#/planet/Kepler-1024_b" target="_blank" rel="noopener">Open NASA Eyes on Exoplanets in a new tab</a>.
</p>
<style>
  .exoplanet-embed {
    width: min(100%, 960px);
    margin: 0 auto;
    background: #05071e;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(3, 7, 40, 0.45);
  }
  .exoplanet-embed iframe {
    border: none;
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    min-height: 360px;
    background: #000;
  }
  .exoplanet-embed-note {
    text-align: center;
    color: #6a6f82;
    font-size: 0.95rem;
    margin: 0.35rem auto 2rem;
  }
  .exoplanet-embed-note a {
    color: #7a67ff;
  }
  @media (max-width: 640px) {
    .exoplanet-embed iframe {
      min-height: 260px;
    }
  }
</style>


### Data 
> The data came from NASA’s exoplanet datasets: KOI (Kepler Object of Interest) and TESS (Transiting Exoplanet Survey Satellite). Each dataset provides rich detail about planetary candidates and their stars, including stellar metrics (temperature, luminosity, radius) and orbital features (period, planet radius, semi-major axis). We built a third, unified dataset to combine the two, allowing us to train the same models on KOI, TESS, and the merged dataset. In total, KOI contains 4,619 records (1,874 candidates; 2,745 confirmed), TESS contains 4,669 records (4,045 candidates; 624 confirmed), and the combined dataset includes 9,308 entries. Additional data features include the Lightkurve API data to recreate the transit method in a visualization.  

<p align="center">
  <img src="https://rogereo.github.io/assets/gifs/koi/3118797.gif" alt="transit" loading="lazy" decoding="async">
</p>

### Algorithm
> We based our modeling approach on two research papers centered on astronomical machine learning. Malik et al. (2022) showed that a LightGBM gradient boosting classifier trained on Kepler and TESS data could achieve near deep learning accuracy (AUC ≈ 0.95 on Kepler, 0.98 on TESS) while remaining efficient and interpretable. Luz et al. (2024) reinforced this by demonstrating that stacking ensembles—which combine multiple models through metalearning consistently outperformed single algorithms on NASA’s KOI dataset. Guided by these findings, we designed a layered approach: four base learners (GradientBoost, XGBoost, AdaBoost, RandomForest) formed the foundation, while LightGBM served as the meta-model to blend their predictions into one refined decision. Each model was trained across three datasets—KOI, TESS, and a combined KOI + TESS configuration—yielding a total of 15 specialized classifiers. This ensemble-first method captures complex, non-linear relationships within tabular astrophysical data while staying computationally efficient and transparent. In essence, our model isn’t a black box—it’s a structured, explainable system built to translate subtle fluctuations of starlight into confident signals of possible new worlds.

### Compute
> All experiments for the Exoplanet Explorer project were conducted locally in a Jupyter Notebook environment within VS Code, using a standard CPU-based laptop—no GPU required. Given the modest size of the KOI and TESS datasets, this lightweight setup was sufficient for training, validation, and visualization without encountering compute bottlenecks. The workflow relied on Python, scikit-learn, and LightGBM, with a 70/15/15 split for training, validation, and testing. The initial plan was to deploy a trained model (saved as a .pkl file) that would allow users to upload exoplanet data and receive live predictions, but due to time constraints, this feature remains under development. Instead, the website currently displays pre-computed prediction results for about 30 exoplanets per dataset—KOI, TESS, and the combined version—showing each exoplanet’s name, ID, individual model outputs (GradientBoost, RandomForest, AdaBoost, XGBoost, and LightGBM), and final ensemble prediction. This approach emphasizes transparency and interpretability, allowing users to see how each model “voted” while keeping computation efficient and entirely accessible through a local-first workflow.

### Evaluation
> To evaluate how well our models could “learn the language of light,” we used AUC (Area Under the ROC Curve) as the primary metric, supported by accuracy, precision, recall, and F1 scores. All experiments followed a 70/15/15 train–validation–test split across five algorithms: GradientBoosting, RandomForest, AdaBoost, XGBoost, and a LightGBM ensemble meta-model. On the KOI dataset, results were stable and consistent, with GradientBoosting leading at an AUC of 0.76—indicating the model could reliably separate candidate from confirmed exoplanets. The TESS dataset, however, proved far noisier: performance dropped to around 0.62 AUC, reflecting challenges like weaker signal-to-noise ratios and class imbalance. When we combined KOI and TESS, the story changed—performance jumped to around 0.84 AUC for RandomForest and XGBoost, showing that uniting data from different missions helped the model generalize better by capturing a wider variety of stellar and orbital patterns. The ensemble LightGBM meta-model achieved a respectable 0.78 AUC but didn’t surpass the strongest individual learners, likely due to overlapping prediction errors among base models. Overall, the results confirmed a key insight echoed in research by Malik et al. (2022) and Luz et al. (2024): tree-based boosting methods remain among the most effective and efficient for exoplanet detection. More importantly, they revealed something poetic—when we listen to multiple signals together, our understanding deepens. The universe may still whisper, but we’re learning to hear its language a little more clearly.

### Deployment
> The initial vision for the Exoplanet Explorer was a live prediction platform where users could upload their own exoplanet data and receive real-time classifications from our trained models. While that interactive backend remains a future goal, deployment evolved into something even more engaging—an immersive, educational experience that brings model results and space data to life. The web application now features an interactive dashboard where users can select between the KOI, TESS, or combined datasets and explore precomputed model results through a blend of scientific visualization and storytelling. Each selection reveals comparative model performance metrics, an interactive 3D t-SNE embedding showing how candidate and confirmed exoplanets cluster in feature space, and a dynamic results table that lists each exoplanet’s name, ID, and predictions from all five models (GradientBoost, RandomForest, AdaBoost, XGBoost, and LightGBM). The dashboard also integrates NASA’s Exoplanet Visualization Tool, allowing users to view a 3D simulation of each planet orbiting its star, compare it to Earth, and explore its orbital path within a solar system analogy. Additional panels visualize light curve animations that show the dip in a star’s brightness during a planetary transit—the very signal our models learned to detect. Together, these elements transform the project from a static model deployment into an interactive learning environment, making the process of exoplanet discovery interpretable, accessible, and deeply human—an invitation to learn the universe’s language of light firsthand.

> Out of interest I added an unsupervised t-SNE model to see how the combined KOI + TESS data would appear in a 3D vector space—revealing how candidate and confirmed exoplanets naturally group based on their shared features.

<div class="responsive-embed" style="width:100%; height:70vh;">
  <iframe src="https://rogereo.github.io/assets/embedding/viewer_comb.html"
          style="border:none; width:100%; height:100%;"></iframe>
</div>


### Conclusion
>In the end, That question captures the curiosity that first drew me toward space exploration—the sense of boundless possibility reflected in the open expanse above. For a long time, I believed that exploring space was something reserved for scientists, engineers, or those already working “in the space.” 

> I was surprised to learn just how much satellite data is publicly available—an entire universe of information open to anyone curious enough to explore it. 

> That weekend transformed my view of space exploration—from something distant and unreachable into something open, collaborative, and profoundly human.

-----
References
- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/) - A World Away: Hunting for Exoplanets with AI
- [Exoplanets](https://science.nasa.gov/exoplanets/) – planets orbiting stars beyond our solar system
- [Transit Method](https://science.nasa.gov/mission/roman-space-telescope/transit-method/) – detecting planets by measuring dips in starlight
- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) – Kepler mission’s catalogue of exoplanet candidates
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI) – TESS mission’s list of planetary candidates
- [LightKurve API](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/using-light-curve-file-products.html?highlight=tesslightcurve&utm_source=chatgpt.com) – Python library for accessing and analyzing NASA light curves
- [Exoplanet Detection Using Machine Learning](https://academic.oup.com/mnras/article/513/4/5505/6472249?login=false) – ML techniques for identifying exoplanet candidates
- [Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification](https://www.mdpi.com/2079-9292/13/19/3950) – study comparing ensemble models for detection
- [Exoplanet Explorer](https://exoplanet-dvu2.onrender.com/) – interactive tool for visualizing exoplanet datasets and model results