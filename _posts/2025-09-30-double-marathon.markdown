---
layout: post
comments: false
title: "Two Marathons, One Weekend"
excerpt: "How can I use technology to help me run both the Cape Town trail and road marathon in one weekend?" 
date:   2025-09-30 00:00:00
mathjax: false
tags: 
  - fitness
  - llm
  - sport
---

### Question
##### How can I use technology to help me run both the Cape Town trail and road marathon in one weekend?

### Why 
> On October 18th and 19th, I am running both the Cape Town [Trail](https://capetowntrailmarathon.com/) and [Road](https://capetownmarathon.com/marathon/) Marathon. 86 kilometers in just 48 hours. Why? The reasons run deeper than chasing finish lines. This challenge is fueled by curiosity, personal growth, and purpose. I’m curious to see how technology can help me prepare for and complete something this demanding, and I believe the process will help me develop habits that carry over into other areas of my life. It’s also about growth, building on past milestones. After completing each race individually in previous years, I finished the trail run alongside another runner. At the finish, the presenter announced that he was still taking on the road marathon the following morning. After eight and a half challenging hours, where do you find the strength to run a full road marathon the next day? I thought "That’s impressive. I wonder if I can do that." Here I am. Most importantly, this effort supports the [Dalen Mmako Foundation (DMF)](https://mmakofoundation.co.za/), a nonprofit started by a close friend who is running alongside me. The DMF creates opportunities for young athletes in disadvantaged communities. DMF not only nurtures sports talent but also uses athletics as a pathway to education and new possibilities—something I connect with deeply. Having benefited from a sports bursary that funded my own education, I know how transformative that foundation can be. This run is my way of uniting personal ambition with a cause that opens doors for the next generation.

### Challenges
> Training for consecutive marathons requires balancing four critical elements: running, strength, recovery, and mindset. My plan is designed to mirror race weekend. Long trail runs on Saturdays, long road runs on Sundays, and steady midweek mileage. It is designed to condition my legs for cumulative fatigue. To stay strong through the training cycle, I’m focusing on gym work that builds leg strength and core stability so I can maintain form and avoid injury. Recovery is equally important and relies on steady nutrition, regular stretching, and a full eight hours of sleep each night for the body to rebuild. The last piece is the mental game. Staying disciplined when motivation fades, managing training alongside work and life, and building resilience to handle both ups and downs. Training, strength, recovery, and mindset are the four pillars that form the foundation for finishing strong.

### Tools

##### AI Running Coach (Running, Strength, & Recovery)
> To take on this challenge I needed a structured plan. I chose not to rely on a generic training plan. Instead, I used a large language model as an AI coach to build a program that balanced running, strength, and recovery. Weekly mileage increased step by step, peaking at 82 kilometers, with long trail runs on Saturdays and road runs on Sundays to replicate race weekend fatigue. On Mondays, Wednesdays, and Fridays I added strength sessions with full-body and core focused work plus short recovery runs. On Tuesdays and Thursdays, 10 km efforts for building endurance capacity. The plan includes lighter baseline weeks and a taper at the end to help me reach race day fresh. I used ChatGPT and Claude in alternation, refining the plan through reflection and iteration until it became my training template. Beyond my own training, I wanted to create something useful for others as a reflection tool in preparing for a race. That led to the [AI Running Coach](https://chatgpt.com/g/g-688a716578588191981e7834f6a464b8-ai-running-coach), a customized tool in the ChatGPT Store. The first MVP version is ready to explore, and it will grow as I continue refining it. 

<div class="table-scroll">
  <table class="training-table">
    <caption>Training Structure (km)</caption>
    <colgroup>
      <col style="min-width: 150px">
      <col span="7" style="min-width: 90px">
      <col style="min-width: 110px">
    </colgroup>
    <thead>
      <tr>
        <th>Week (Dates)</th>
        <th>Mon<br><small>Strength + 4k</small></th>
        <th>Tue<br><small>10k</small></th>
        <th>Wed<br><small>Strength + 4k</small></th>
        <th>Thu<br><small>10k</small></th>
        <th>Fri<br><small>Core + 4k</small></th>
        <th>Sat<br><small>Trail LR</small></th>
        <th>Sun<br><small>Road LR</small></th>
        <th>Weekly Total</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>(Sept 8–14)</td><td>4</td><td>10</td><td>4</td><td>10</td><td>4</td><td>12</td><td>14</td><td>58</td>
      </tr>
      <tr>
        <td>(Sept 15–21)</td><td>4</td><td>10</td><td>4</td><td>10</td><td>4</td><td>16</td><td>18</td><td>66</td>
      </tr>
      <tr>
        <td>(Sept 22–28)</td><td>4</td><td>10</td><td>4</td><td>10</td><td>4</td><td>20</td><td>22</td><td>74</td>
      </tr>
      <tr>
        <td>(Sept 29–Oct 5)</td><td>4</td><td>10</td><td>4</td><td>10</td><td>4</td><td>24</td><td>26</td><td>82</td>
      </tr>
      <tr>
        <td>(Oct 6–12)</td><td>4</td><td>10</td><td>4</td><td>10</td><td>4</td><td>14</td><td>16</td><td>62</td>
      </tr>
      <tr>
        <td>(Race Week Oct 13–19)</td><td>4</td><td>Rest</td><td>4</td><td>Rest</td><td>4</td><td>Trail 42</td><td>Road 42</td><td>—</td>
      </tr>
    </tbody>
  </table>
</div>

<style>
/* minimal, readable table styles (scoped to this post) */
.table-scroll { margin: 0.75rem 0 1.25rem; }
.training-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;                   /* distribute columns evenly */
  font-size: 0.95rem;
  line-height: 1.4;
  border: 1px solid #e5e7eb;             /* light gray */
}
.training-table caption {
  caption-side: top;
  text-align: left;
  font-weight: 700;
  margin-bottom: .5rem;
  color: #111827;                        /* near-black */
}
.training-table thead th {
  background: #f9fafb;                   /* very light gray */
  color: #111827;
  text-align: center;
  font-weight: 700;
  padding: .6rem .5rem;
  border-bottom: 1px solid #e5e7eb;
  word-wrap: break-word;                 /* allow wrapping */
}
.training-table tbody td, .training-table tbody th {
  padding: .55rem .5rem;
  text-align: center;
  border-top: 1px solid #f1f5f9;         /* row separators */
  word-wrap: break-word;
}
.training-table tbody tr:nth-child(even) td {
  background: #fcfcfd;                   /* subtle zebra */
}
.training-table tbody td:first-child,
.training-table thead th:first-child {
  text-align: left;
  font-weight: 600;
  width: 18%;                            /* slightly wider for dates */
}
.training-table small { font-weight: 500; color: #6b7280; } /* muted labels */

/* compact on small screens */
@media (max-width: 560px) {
  .training-table { font-size: 0.88rem; }
  .training-table thead th { padding: .5rem .4rem; }
  .training-table tbody td { padding: .45rem .4rem; }
}
</style>

##### Beeminder (Running & Mindset)
> To stay accountable, I designed a two-part system with Strava and [Beeminder](https://help.beeminder.com/article/70-what-is-beeminder)
. Strava tracks my runs in real time, and Beeminder converts that data into a contract with real consequences. 
I came across Beeminder in Nick Winter’s book [The Motivation Hacker](https://www.nickwinter.net/the-motivation-hacker). Instead of relying on willpower, you link your commitment to real consequences. Miss a goal, and you pay — literally. My target averages out to 6 kilometers per day, enough to stay on pace for the double marathon. With each run syncing automatically, Beeminder keeps me accountable through a simple graph: green when I’m on track, red when I’m slipping. If I fall short, the pledge doubles ($5 becomes $10, then $20 etc). The twist? I haven’t lost money. I find that it keeps me accountable. On days I felt like skipping, the thought of losing money pushed me out the door. What makes it powerful is that it feels like a game. Suddenly, training feels fun. It’s counterintuitive, but that’s the secret to why it works.


<!-- beeminder header + graph (via cloudflare worker; no token exposed) -->
<div id="bm"
     data-user="roger_a11"
     data-goal="doublemarathonprep"
     data-title="doublemarathonprep"></div>

<style>
  .bm-row{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}
  .bm-link{font-weight:700;text-decoration:underline}
  .bm-chip{padding:.28rem .56rem;border-radius:.4rem;font-weight:700;line-height:1}
  .bm-buf{background:#333;color:#ffc107}
  .bm-ctd{background:#16a34a;color:#fff}
  .bm-plg{background:#444;color:#ffc107}
  .bm-img{max-width:100%;height:auto;border:0}
</style>

<script>
(async () => {
  const el = document.getElementById('bm');
  const user = el?.dataset.user, goal = el?.dataset.goal;
  const title = el?.dataset.title;

  const worker_base = 'https://beeminder-proxy.roger-arendse713.workers.dev';
  const api  = `${worker_base}/bm/${user}/${goal}`;
  const page = `https://www.beeminder.com/${user}/${goal}`;

  let g = null;
  try { const r = await fetch(api); if (r.ok) g = await r.json(); } catch(_) {}
  // Helpful for debugging the fields coming back from your worker:
  console?.log?.('beeminder payload', g);

  const fmt = ms => ms<=0 ? 'due now'
    : `${Math.floor(ms/86400000)}d ${Math.floor(ms%86400000/3600000)}h ${Math.floor(ms%3600000/60000)}m`;

  const row = document.createElement('div'); row.className='bm-row';
  const link = document.createElement('a'); link.className='bm-link';
  link.href = page; link.target = '_blank'; link.rel = 'noopener';
  link.textContent = title || `${user}/${goal}`;

  const buf=document.createElement('span'); buf.className='bm-chip bm-buf';
  const ctd=document.createElement('span'); ctd.className='bm-chip bm-ctd';
  const plg=document.createElement('span'); plg.className='bm-chip bm-plg';

  // countdown → losedate
  const losedate = g?.losedate ? g.losedate * 1000 : null;
  const tick = () => {
    if (!losedate) { ctd.textContent = '—'; return; }
    const ms = Math.max(0, losedate - Date.now());
    ctd.textContent = fmt(ms);
  };
  tick(); if (losedate) setInterval(tick, 60000);

  // buffer → prefer delta/rate (km / (km/day) = days). Fallback to g.safebuf only if needed.
  const rate  = (typeof g?.rate  === 'number') ? Math.abs(g.rate)  : null;
  const delta = (typeof g?.delta === 'number') ? g.delta : null;

  let bufDays = null;
  if (rate && rate > 0 && typeof delta === 'number') {
    bufDays = delta / rate;                         // correct definition
  } else if (typeof g?.safebuf === 'number') {
    bufDays = g.safebuf;                            // fallback if proxy supplies it correctly
  }

  buf.textContent = (typeof bufDays === 'number' && isFinite(bufDays))
    ? `+${bufDays.toFixed(4)}`
    : '+—';

  // pledge
  plg.textContent = (g && g.pledge != null) ? `$${g.pledge}` : '$—';

  row.append(link, document.createTextNode(':'), buf,
             document.createTextNode(' due in '), ctd,
             document.createTextNode(' or pay '), plg);

  const img = document.createElement('img');
  img.className='bm-img'; img.src=`${page}.png`; img.alt='beeminder goal graph';

  el.replaceChildren(row, img);
})();
</script>


##### Feedback System (Running, Recovery & Mindset)

> I came across this habit tracking system on Pinterest. It inspired me to create my own customized version to serve as a tool for tracking my progress. an Input-Output Log

<p align="center">
  <img src="/assets/tracker.png" alt="tracker" loading="lazy" decoding="async" style="width:60%; height:auto;">
</p>

> A tool for logging nutrition intake and tracking output through training and recovery work. As part of this, I track my body weight every morning and evening. The intensity of training means my body will naturally shed weight. I want to stay ahead of this by maintaining my current body structure throughout training. By tracking my weight, I can adjust my nutrition whenever it dips. It serves as a feedback loop. I added another graph to track my Body State Index, a subjective metric that reflects how I feel each day. The metrics include sleep, energy, and flexibility. How well did I sleep? What were my energy levels? How flexible did I feel today? Sleep is logged in the morning, while energy and flexibility are tracked at the end of the day. I think of it as an experiment to see how feelings connect with data over time. Here is the [template](https://docs.google.com/spreadsheets/d/1muXk-82C541gbJ1P-6h9snPo2eWTSVh8RIfgqkYkqw8/edit?usp=sharing) if you’d like to try it out or modify it into something that works for you.

<p align="center">
  <img src="/assets/IO_Log_Sept30_2025.jpg" alt="IO_Log" loading="lazy" decoding="async">
</p>

> To make things more engaging, I started treating this log like raw data for a coach. Feeding it into ChatGPT turns the numbers into feedback, almost like having a second set of eyes on my training. Here’s the latest analysis

<p align="center">
  <img src="/assets/GPT_Response.png" alt="GPT" loading="lazy" decoding="async" style="width:60%; height:auto;">
</p>

### Final Thoughts
Two marathons in one weekend is the challenge. The real story is how technology helps turn curiosity into growth, and ambition into action.

----
References
- [Dalen Mmako Foundation](https://mmakofoundation.co.za/) – nonprofit supporting young athletes in disadvantaged communities
- [AI Running Coach](https://chatgpt.com/g/g-688a716578588191981e7834f6a464b8-ai-running-coach) – my custom GPT experiment for marathon training
- [Input-Output Log Template](https://docs.google.com/spreadsheets/d/1muXk-82C541gbJ1P-6h9snPo2eWTSVh8RIfgqkYkqw8/edit?usp=sharing) – spreadsheet I use to track nutrition, weight, and body state  
- [Beeminder](https://help.beeminder.com/article/70-what-is-beeminder) – habit tracking tool that integrates with Strava
- [Strava](https://www.strava.com/athletes/91022371) – follow my training progress

