#!/usr/bin/env python3
"""Build a self-contained reveal.js HTML presentation for lab meeting.

Reads 8 figure PNGs, converts to base64, and generates a single HTML file
with all images embedded inline. No external dependencies beyond the CDN.
"""

import base64
import os
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
OUTPUT_FILE = Path(__file__).parent / "lab_meeting_presentation.html"

FIGURE_FILES = {
    "fig1": "Figure1_pipeline.png",
    "fig2": "figure2_dataset_overview.png",
    "fig3": "figure3_synthetic_validation.png",
    "fig4": "figure4_truncation_comparison.png",
    "fig5": "figure5_haldane_vs_gompertz.png",
    "fig6": "Figure6_Ki_forest_plot.png",
    "fig7": "Figure7_operator_reproducibility.png",
    "fig8": "Figure8_representative_curves.png",
}


def load_figure_b64(filename: str) -> str:
    """Read a PNG file and return its base64-encoded data URI."""
    path = FIGURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing figure: {path}")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    print(f"  Encoded {filename} ({path.stat().st_size / 1024:.0f} KB)")
    return f"data:image/png;base64,{data}"


def build_html(figures: dict[str, str]) -> str:
    """Generate the full HTML presentation."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bacterial Growth Curve Pipeline &mdash; Lab Meeting</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/white.css">
<style>
  :root {{
    --accent: #0077b6;
    --accent-light: #90e0ef;
    --dark: #1a1a2e;
  }}
  .reveal h1, .reveal h2, .reveal h3 {{
    color: var(--accent);
    font-family: 'Helvetica Neue', Arial, sans-serif;
    text-transform: none;
    letter-spacing: -0.02em;
  }}
  .reveal h1 {{ font-size: 1.6em; line-height: 1.2; }}
  .reveal h2 {{ font-size: 1.3em; margin-bottom: 0.4em; }}
  .reveal h3 {{ font-size: 1.0em; }}
  .reveal section {{ text-align: left; }}
  .reveal ul {{ font-size: 0.78em; line-height: 1.5; }}
  .reveal li {{ margin-bottom: 0.35em; }}
  .reveal p {{ font-size: 0.78em; line-height: 1.5; }}
  .reveal img.figure {{
    max-height: 55vh;
    max-width: 90%;
    display: block;
    margin: 0.5em auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
  }}
  .reveal .title-slide {{ text-align: center; }}
  .reveal .title-slide h1 {{ font-size: 1.35em; margin-bottom: 0.3em; }}
  .reveal .title-slide p {{ font-size: 0.7em; margin: 0.15em 0; color: #555; }}
  .reveal .title-slide .authors {{ font-size: 0.8em; color: #333; margin-top: 0.5em; }}
  .reveal .title-slide .affiliation {{ font-size: 0.65em; color: #666; }}
  .reveal strong {{ color: var(--accent); }}
  .reveal .metric-box {{
    background: #f0f7ff;
    border-left: 4px solid var(--accent);
    padding: 0.5em 1em;
    margin: 0.5em 0;
    border-radius: 4px;
    font-size: 0.78em;
  }}
  .reveal .metric-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6em;
    margin-top: 0.5em;
  }}
  .reveal .highlight-num {{
    font-size: 1.4em;
    font-weight: bold;
    color: var(--accent);
  }}
  .reveal .questions-slide {{ text-align: center; }}
  .reveal .questions-slide h2 {{ font-size: 2em; margin-bottom: 0.5em; }}
  .reveal .slide-number {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="reveal">
<div class="slides">

<!-- SLIDE 1: Title -->
<section class="title-slide">
  <h1>Automated Computational Pipeline for Bacterial Growth Curve Classification and Substrate Inhibition Modeling in Pesticide Bioremediation Screening</h1>
  <p class="authors">Daniel Sambold, Mary Snook, Nate Walton, Dominique Scott, and Nathaniel Jue</p>
  <p class="affiliation">Jue Lab &mdash; Department of Marine Science<br>California State University, Monterey Bay</p>
  <p class="affiliation">April 2026</p>
  <aside class="notes">Thank you for having me. Today I'm going to present our automated pipeline for analyzing bacterial growth curves in the context of pesticide bioremediation screening.</aside>
</section>

<!-- SLIDE 2: The Problem -->
<section>
  <h2>The Problem</h2>
  <ul>
    <li>Bioremediation screening generates <strong>hundreds of growth curves</strong> per experiment</li>
    <li>Manual analysis: slow, subjective, non-reproducible across operators</li>
    <li>Existing tools (Growthcurver, grofit, AMiGA) lack:
      <ul>
        <li>ML-based classification</li>
        <li>Substrate inhibition kinetics</li>
        <li>Bayesian uncertainty quantification</li>
        <li>Inter-operator comparison</li>
      </ul>
    </li>
    <li>We need: <strong>automated, validated, operator-independent analysis</strong></li>
  </ul>
  <aside class="notes">TECAN plate readers generate dense time-series data, but analyzing it is still a bottleneck. Existing tools fit growth models but don't classify curve quality, don't model substrate inhibition, and can't compare across operators.</aside>
</section>

<!-- SLIDE 3: Dataset Overview -->
<section>
  <h2>Dataset Overview</h2>
  <ul>
    <li><strong>161</strong> averaged growth curves from 6 experimental groups</li>
    <li><strong>3</strong> independent operators spanning 2018&ndash;2025</li>
    <li><strong>7</strong> pesticides &mdash; 3 chemical classes (neonicotinoids, organophosphates, pyrethroids)</li>
    <li><strong>66 curves</strong> classified as GOOD (41%)</li>
  </ul>
  <img class="figure" src="{figures['fig2']}" alt="Figure 2: Dataset overview">
  <aside class="notes">Our dataset comes from three operators over a seven-year period. We tested seven pesticides across three chemical classes. About 41% of curves showed genuine growth &mdash; the rest were noise, contamination, or failed inoculations.</aside>
</section>

<!-- SLIDE 4: Pipeline Architecture -->
<section>
  <h2>Pipeline Architecture</h2>
  <ul>
    <li><strong>11 automated steps</strong> (0&ndash;10): preprocessing &rarr; ML classification &rarr; Gompertz fitting &rarr; Haldane inhibition &rarr; Bayesian analysis &rarr; operator comparison</li>
    <li>Configuration-driven, fully reproducible</li>
    <li>~80 minutes for complete analysis</li>
  </ul>
  <img class="figure" src="{figures['fig1']}" alt="Figure 1: Pipeline schematic">
  <aside class="notes">The pipeline has 11 steps that run automatically. Everything is config-driven &mdash; you point it at your raw data and it produces publication-ready results. The bottleneck is the Bayesian Haldane sampling at about 40 minutes.</aside>
</section>

<!-- SLIDE 5: Gompertz Model + Truncation -->
<section>
  <h2>Gompertz Model + Adaptive Truncation</h2>
  <ul>
    <li>Modified Gompertz: <em>y(t) = A &middot; exp(&minus;exp((&mu;e/A)(&lambda;&minus;t)+1))</em></li>
    <li>Parameters: <strong>A</strong> (max density), <strong>&mu;</strong> (growth rate), <strong>&lambda;</strong> (lag time)</li>
    <li>MCCV truncation removes death-phase artifacts</li>
    <li><strong>7 truncation methods compared</strong>; MCCV selected as optimal</li>
  </ul>
  <img class="figure" src="{figures['fig4']}" alt="Figure 4: Truncation comparison">
  <aside class="notes">We fit the modified Gompertz model to each curve. A key innovation is adaptive truncation &mdash; we remove the death phase so it doesn't corrupt the growth rate estimate. We compared seven truncation strategies and MCCV performed best.</aside>
</section>

<!-- SLIDE 6: Two-Stage ML Classification -->
<section>
  <h2>Two-Stage ML Classification</h2>
  <div class="metric-grid">
    <div class="metric-box">
      <strong>Stage 1: Pre-fit gate</strong><br>
      Rejects obvious junk before fitting (saves compute)
    </div>
    <div class="metric-box">
      <strong>Stage 2: Post-fit classifier</strong><br>
      24 engineered features, gradient boosting
    </div>
    <div class="metric-box">
      Held-out accuracy<br>
      <span class="highlight-num">99.5%</span><br>
      on 175 test curves
    </div>
    <div class="metric-box">
      Precision<br>
      <span class="highlight-num">100%</span><br>
      Zero false positives
    </div>
  </div>
  <ul>
    <li>Continuous confidence score enables <strong>borderline review</strong></li>
  </ul>
  <aside class="notes">The ML classifier works in two stages. The pre-fit gate catches obviously bad curves before we waste time fitting them. The post-fit classifier uses 24 features including fit quality metrics and derived ratios. It achieved 99.5% accuracy with perfect precision &mdash; it never called a bad curve good.</aside>
</section>

<!-- SLIDE 7: Independent Validation -->
<section>
  <h2>Independent Validation</h2>
  <ul>
    <li><strong>555</strong> synthetic curves, different random seed</li>
    <li>Completely independent of training data &mdash; <strong>no data leakage</strong></li>
    <li><strong>98.7% accuracy</strong> (388 TP, 160 TN, 5 FP, 2 FN)</li>
    <li>7 misclassifications in borderline/edge cases</li>
  </ul>
  <img class="figure" src="{figures['fig3']}" alt="Figure 3: Synthetic validation">
  <aside class="notes">To get honest validation numbers, we generated a completely independent set of 555 synthetic curves using a different random seed. The classifier achieved 98.7% accuracy. The seven failures were all borderline cases where even manual inspection would be difficult.</aside>
</section>

<!-- SLIDE 8: Parameter Recovery -->
<section>
  <h2>Parameter Recovery</h2>
  <div class="metric-grid">
    <div class="metric-box">
      <strong>&lambda; (lag time)</strong><br>
      <span class="highlight-num">R&sup2; = 0.966</span><br>
      Excellent recovery
    </div>
    <div class="metric-box">
      <strong>&mu; (growth rate)</strong><br>
      <span class="highlight-num">R&sup2; = 0.751</span><br>
      Good recovery
    </div>
    <div class="metric-box">
      <strong>A (carrying capacity)</strong><br>
      <span class="highlight-num">R&sup2; &lt; 0</span><br>
      Poor (expected)
    </div>
    <div class="metric-box">
      <strong>Trade-off</strong><br>
      Truncation sacrifices A accuracy for robust &mu; and &lambda;
    </div>
  </div>
  <ul>
    <li>Honest limitation: <strong>A estimates should be used with caution</strong></li>
  </ul>
  <aside class="notes">Parameter recovery shows lag time is recovered almost perfectly, growth rate is good, but carrying capacity is poor. This is an inherent trade-off &mdash; truncation removes the plateau that A depends on. We're transparent about this limitation.</aside>
</section>

<!-- SLIDE 9: Haldane Substrate Inhibition -->
<section>
  <h2>Haldane Substrate Inhibition</h2>
  <ul>
    <li>Haldane ODE: d<em>X</em>/d<em>t</em> = &mu;(<em>S</em>)&middot;<em>X</em>&middot;(1&minus;<em>X</em>/<em>X</em><sub>max</sub>)</li>
    <li>where &mu;(<em>S</em>) = &mu;<sub>max</sub>&middot;<em>S</em> / (<em>K<sub>s</sub></em> + <em>S</em> + <em>S</em>&sup2;/<em>K<sub>i</sub></em>)</li>
    <li><strong>K<sub>i</sub></strong> = inhibition constant (lower = more inhibitory)</li>
    <li>Haldane preferred over Gompertz in <strong>30/42 cases (71%)</strong></li>
    <li>Model selection via corrected AIC</li>
  </ul>
  <img class="figure" src="{figures['fig5']}" alt="Figure 5: Haldane vs Gompertz">
  <aside class="notes">Beyond simple growth curves, we fit Haldane substrate inhibition models to capture how pesticide concentration affects growth. The Haldane model was preferred 71% of the time, confirming that substrate inhibition is a real feature of these systems.</aside>
</section>

<!-- SLIDE 10: Bayesian Ki Estimates -->
<section>
  <h2>Bayesian K<sub>i</sub> Estimates</h2>
  <ul>
    <li>Hierarchical Bayesian model with partial pooling</li>
    <li><strong>Imidacloprid: K<sub>i</sub> = 4.83</strong> (most inhibitory)</li>
    <li>Flupyradifurone: K<sub>i</sub> = 5.87</li>
    <li>Bifenthrin: K<sub>i</sub> = 17.55 (least inhibitory)</li>
    <li>Full posterior distributions enable uncertainty quantification</li>
  </ul>
  <img class="figure" src="{figures['fig6']}" alt="Figure 6: Bayesian Ki forest plot">
  <aside class="notes">The Bayesian analysis gives us full posterior distributions for the inhibition constants. Imidacloprid had the lowest Ki, making it the most inhibitory pesticide. Bifenthrin was the least inhibitory. The wide credible intervals for some pesticides reflect limited sample sizes.</aside>
</section>

<!-- SLIDE 11: Inter-Operator Reproducibility -->
<section>
  <h2>Inter-Operator Reproducibility</h2>
  <ul>
    <li>3 operators, 5 shared imidacloprid strains</li>
    <li><strong>0/12 ANOVA tests significant</strong></li>
    <li>But coefficient of variation = <strong>61.8%</strong></li>
    <li>Pipeline produces consistent classifications despite biological variability</li>
  </ul>
  <img class="figure" src="{figures['fig7']}" alt="Figure 7: Operator reproducibility">
  <aside class="notes">We compared growth parameters across three operators using five shared strains. None of the statistical tests were significant, but the coefficient of variation was 61.8%, showing real biological variability. The pipeline itself is reproducible &mdash; the variability comes from the biology.</aside>
</section>

<!-- SLIDE 12: Representative Curves -->
<section>
  <h2>Representative Curves</h2>
  <img class="figure" src="{figures['fig8']}" alt="Figure 8: Representative curve fits" style="max-height: 65vh;">
  <aside class="notes">Here are some representative curves showing how the pipeline handles different scenarios &mdash; clean growth, noisy data, curves that needed truncation.</aside>
</section>

<!-- SLIDE 13: Key Takeaways -->
<section>
  <h2>Key Takeaways</h2>
  <ul>
    <li>Fully automated, <strong>11-step pipeline</strong> for TECAN data</li>
    <li><strong>98.7%</strong> validated classification accuracy (no data leakage)</li>
    <li>Imidacloprid and flupyradifurone <strong>most inhibitory</strong> (lowest K<sub>i</sub>)</li>
    <li>Haldane kinetics preferred in <strong>71%</strong> of cases</li>
    <li>Honest about limitations: A recovery, 41% good rate</li>
    <li>Reproducible across operators and years</li>
  </ul>
  <aside class="notes">To summarize &mdash; we built a fully automated pipeline that handles everything from raw plate reader data to publication-ready results. The validation is honest and the limitations are documented.</aside>
</section>

<!-- SLIDE 14: Future Directions -->
<section>
  <h2>Future Directions</h2>
  <ul>
    <li>Expand to additional bacterial isolates and pesticides</li>
    <li>Concentration gradient experiments for refined K<sub>i</sub> estimates</li>
    <li>Field-scale validation of bioremediation candidates</li>
    <li>Community adoption &mdash; open-source Python pipeline</li>
    <li>Integration with other plate reader formats beyond TECAN</li>
  </ul>
  <aside class="notes">Going forward, we want to test more isolates, do proper concentration gradients for better Ki estimates, and validate our top candidates in the field.</aside>
</section>

<!-- SLIDE 15: Questions -->
<section class="questions-slide">
  <h2>Questions?</h2>
  <p style="margin-top: 1em; font-size: 0.7em; color: #555;">
    <strong>Acknowledgments:</strong> Jue Lab, CSUMB Department of Marine Science<br><br>
    <strong>Funded by:</strong> [to be filled]<br><br>
    <strong>Code available at:</strong> [to be filled]
  </p>
  <aside class="notes">Thank you. I'm happy to take any questions.</aside>
</section>

</div><!-- /.slides -->
</div><!-- /.reveal -->

<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/notes/notes.js"></script>
<script>
  Reveal.initialize({{
    hash: true,
    slideNumber: true,
    width: 1280,
    height: 720,
    margin: 0.08,
    plugins: [ RevealNotes ]
  }});
</script>
</body>
</html>"""


def main():
    print("Building lab meeting presentation...")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Output file: {OUTPUT_FILE}\n")

    # Load all figures
    print("Encoding figures:")
    figures = {}
    for key, filename in FIGURE_FILES.items():
        figures[key] = load_figure_b64(filename)

    # Generate HTML
    print("\nGenerating HTML...")
    html = build_html(figures)

    # Write output
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Written: {OUTPUT_FILE}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Slides: 15")
    print("\nDone! Open the HTML file in a browser. Press S for speaker notes.")


if __name__ == "__main__":
    main()
