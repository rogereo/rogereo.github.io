---
layout: post
comments: false
title: "Two Marathons, One Weekend (Take Two)"
excerpt: "How can I use technology to run both the Cape Town trail and road marathon in one weekend, the second time around?"
date:   2026-05-19 00:00:00
mathjax: false
tags:
  - fitness
  - llm
  - quantified-self
---

### Question
##### How can I use technology to run both the Cape Town trail and road marathon in one weekend, the second time around?

### Why

> Last October, I lined up for the Cape Town [Trail](https://capetowntrailmarathon.com/) and [Road](https://capetownmarathon.com/marathon/) Marathon in the same weekend. 86 kilometers in 48 hours. I wrote about it [here](https://rogereo.github.io/2025/09/30/double-marathon/). The trail run went ahead. The road marathon was cancelled. The challenge stayed unfinished. On May 23rd and 24th, I am attempting it again. Same idea. Same distance. Same cause. The reasons have not changed: curiosity about what technology can do for a body that has to recover overnight, growth from putting myself in the path of something hard, and purpose through running for the [Dalen Mmako Foundation (DMF)](https://mmakofoundation.co.za/), which uses sport as a pathway to education for young people in disadvantaged communities. The first attempt taught me that the system around the run matters more than the run itself. This time I am keeping what worked, dropping what did not, and adding one thing I should have had from the start.

### Challenges

> The four pillars are still the same: running, strength, recovery, and mindset. What is different is what I now know about each of them. From the first attempt I learned that cumulative fatigue is less about the long runs and more about what happens between them. Recovery is where the race is actually won or lost. The mental side is also different the second time around. There is a quiet pressure that comes from attempting something you have already publicly committed to once. The training cycle is shorter, the build is steeper, and the margin for error is smaller. The plan has to account for all of it.

### Tools

##### Custom Training Schedule (Running, Strength, & Recovery)

> I used Claude to build a custom training schedule for this cycle, shaped around a May race weekend and what I learned the first time. I used it as a framework rather than a script. The structure is similar to last year: long trail runs on Saturdays, long road runs on Sundays, midweek mileage, gym work, and a taper into race week. What changed is the shape of the build. The cycle is shorter (six weeks instead of six and a half), Week 4 is a deliberate recovery dip before Week 5 peaks at 78 km, and the weekend pairings are tuned closer to what race day actually looks like. Recovery first, mileage second.

<div class="table-scroll">
  <table class="training-table">
    <caption>Training Structure (km)</caption>
    <thead>
      <tr>
        <th>Week (Dates)</th>
        <th>Mon<br><small>Gym + 4k</small></th>
        <th>Tue<br><small>10k</small></th>
        <th>Wed<br><small>10k</small></th>
        <th>Thu<br><small>Gym + 4k</small></th>
        <th>Fri<br><small>Gym + 4k</small></th>
        <th>Sat<br><small>Trail LR</small></th>
        <th>Sun<br><small>Road LR</small></th>
        <th>Weekly Total</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>(Apr 13–19)</td><td>4</td><td>10</td><td>10</td><td>4</td><td>4</td><td>15</td><td>12</td><td>59</td></tr>
      <tr><td>(Apr 20–26)</td><td>4</td><td>10</td><td>10</td><td>4</td><td>4</td><td>18</td><td>15</td><td>65</td></tr>
      <tr><td>(Apr 27–May 3)</td><td>4</td><td>10</td><td>10</td><td>4</td><td>4</td><td>22</td><td>18</td><td>72</td></tr>
      <tr><td>(May 4–10, Recovery)</td><td>4</td><td>10</td><td>10</td><td>4</td><td>4</td><td>16</td><td>14</td><td>62</td></tr>
      <tr><td>(May 11–17, Peak)</td><td>4</td><td>10</td><td>10</td><td>4</td><td>4</td><td>25</td><td>21</td><td>78</td></tr>
      <tr><td>(Race Week May 18–24)</td><td>4</td><td>Rest</td><td>4 easy</td><td>Rest</td><td>Rest</td><td>Trail 43</td><td>Road 42</td><td>—</td></tr>
    </tbody>
  </table>
</div>

<style>
/* minimal, readable table styles (scoped to this post) */
.table-scroll { margin: 0.75rem 0 1.25rem; }
.training-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 0.95rem;
  line-height: 1.4;
  border: 1px solid #e5e7eb;
}
.training-table caption {
  caption-side: top;
  text-align: left;
  font-weight: 700;
  margin-bottom: .5rem;
  color: #111827;
}
.training-table thead th {
  background: #f9fafb;
  color: #111827;
  text-align: center;
  font-weight: 700;
  padding: .6rem .5rem;
  border-bottom: 1px solid #e5e7eb;
  word-wrap: break-word;
}
.training-table tbody td,
.training-table tbody th {
  padding: .55rem .5rem;
  text-align: center;
  border-top: 1px solid #f1f5f9;
  word-wrap: break-word;
}
.training-table tbody tr:nth-child(even) td { background: #fcfcfd; }
.training-table tbody td:first-child,
.training-table thead th:first-child {
  text-align: left;
  font-weight: 600;
  width: 18%;
}
.training-table small { font-weight: 500; color: #6b7280; }
@media (max-width: 560px) {
  .training-table { font-size: 0.88rem; }
  .training-table thead th { padding: .5rem .4rem; }
  .training-table tbody td { padding: .45rem .4rem; }
}
</style>

##### Beeminder (Running & Mindset)

> The accountability system is the same one I described in the first post, with one change: the daily target is higher. Strava tracks my runs in real time, and [Beeminder](https://help.beeminder.com/article/70-what-is-beeminder) turns that data into a contract with real money on the line. I came across Beeminder in Nick Winter's book [The Motivation Hacker](https://www.nickwinter.net/the-motivation-hacker). Instead of relying on willpower, you link the commitment to tangible stakes. Miss the goal, the pledge doubles ($5 becomes $10, then $20, and so on). My target this cycle averages 9 kilometers per day, up from 6 last time, which reflects the shorter runway to race weekend. Each run syncs automatically. The graph turns green when I am on track and red when I am slipping. The mechanic is still the same one that surprised me last year: on days I felt like skipping, the thought of losing money pushed me out the door. What makes it powerful is that it feels like a game. Suddenly, training feels fun. It’s counterintuitive, but that’s the secret to why it works. You can follow along on the [goal page](https://www.beeminder.com/roger_a11/doublemarathon).

<!-- beeminder header + graph (via cloudflare worker; no token exposed) -->
<div id="bm"
     data-user="roger_a11"
     data-goal="doublemarathon"
     data-title="roger_a11/doublemarathon"></div>

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
  try {
    const r = await fetch(api);
    if (r.ok) g = await r.json();
    else console?.warn?.('beeminder worker non-ok', r.status, r.statusText);
  } catch(e) { console?.warn?.('beeminder worker fetch failed', e); }
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

  // graph: prefer the rich annotated graph_url from the API payload (bullseye, watermark, etc).
  // Fall back to the bare public PNG only if the payload doesn't include one.
  const graphSrc = (g && (g.graph_url || g.thumb_url)) || `${page}.png`;
  const img = document.createElement('img');
  img.className='bm-img'; img.src = graphSrc; img.alt='beeminder goal graph';

  el.replaceChildren(row, img);
})();
</script>

##### Sleep & Metrics Dashboard (Recovery)

<p align="center">
  <img src="/assets/garmin-dashboard.png" loading="lazy" decoding="async">
</p>

> This is the new piece. Last cycle, I tracked recovery subjectively through the Body State Index in my Input-Output Log. It worked, but it was noisy. A number I gave myself in the morning was a feeling, not a measurement. This time I wanted something more systematic, so I built a custom dashboard that pulls data from my Garmin watch and surfaces the metrics that matter for training. Sleep is the headline section: total sleep duration plotted against an 8-hour goal line, with the Garmin sleep score below it. A 7-day rolling average smooths the daily noise so I can see the trend rather than chase any single bad night. The dashboard also has sections for activity and steps, heart and respiration, stress and body battery, and nutrition. The point is not to optimize any single number. It is to catch the patterns early: a week of compressed sleep before I feel the fatigue, a stress trend that says today should be easy regardless of what the plan says, a nutrition gap that explains why the long run felt heavier than it should have. Recovery was the weakest link in the first attempt. This dashboard is how I am trying to close that gap.

<p align="center">
  <img src="/assets/sleep-score.png" loading="lazy" decoding="async">
</p>

> The Apr 11 and Apr 24 dips are the kind of bad nights I would not have logged honestly in the morning. The dashboard catches what a feeling misses.

<p align="center">
  <img src="/assets/sleep-stages.png" loading="lazy" decoding="async">
</p>

> Apr 19 stands out: nearly 14 hours after a tough weekend. The body asked for it.

<p align="center">
  <img src="/assets/stage-composition.png" loading="lazy" decoding="async">
</p>

> Deep sleep hovers at 11–15%, the bottom of the typical 13–23% range. Enough hours, not enough depth. That is the gap to close.

### Final Thoughts

> The first attempt left a question hanging. Can I actually do this, or did I just do half of it? The second attempt is the answer. Same challenge, sharper system. Two marathons in one weekend is still the goal. This time, I am running the system as much as the distance.

-----
References
- [Dalen Mmako Foundation](https://mmakofoundation.co.za/) - nonprofit supporting young athletes in disadvantaged communities
- [First attempt blog post](https://rogereo.github.io/2025/09/30/double-marathon/) - Two Marathons, One Weekend (2025)
- [Beeminder goal](https://www.beeminder.com/roger_a11/doublemarathon) - current double marathon commitment
- [Garmin Dashboard repo](https://github.com/rogereo/garmin-dashboard) - clone, add your Garmin login to the .env, and run the same view on your own data (requires a Garmin watch synced via the Garmin Connect app)
- [Strava](https://www.strava.com/athletes/91022371) - training progress