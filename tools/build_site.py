#!/usr/bin/env python3
"""Build the README and dependency-free GitHub Pages site from the same editorial source."""
from pathlib import Path
from html import escape
import json
import re
import shutil
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website"
DOCS = ROOT / "docs"
REPO = "https://github.com/CerulloE1996/NicoStan"
## Absolute site address: Facebook/Twitter link previews need full https URLs for og:url and og:image (relative paths fail).
SITE_URL = "https://cerulloe1996.github.io/NicoStan/"
DOCS.mkdir(exist_ok=True)
(DOCS / "assets").mkdir(exist_ok=True)
## The page of the full code of every example model and script (built at the end of this file); the website's links
## to a file in inst/examples open that file's panel on it.
EXAMPLES = ROOT / "inst" / "examples"
EXAMPLE_PAGE = "example_models_full_code.html"


def anchor_of_example_file(relative_path):
    return "file-" + re.sub(r"[^A-Za-z0-9]+", "-", relative_path).strip("-").lower()
algorithms = json.loads((SOURCE / "algorithms.json").read_text())
references = json.loads((SOURCE / "references.json").read_text())

quickstart = '''```r
source(system.file("examples", "random_intercepts.R", package = "NicoStan"))
fit <-  run_random_intercepts(burnin_algorithm = "CHESSR")
diagnostics <-  fit$summary(save_log_lik_trace = FALSE)
fit$model_fit_object$summaries$summary_tibbles$summary_tibble_main_params

## Other trajectory-length adaptation options:
## "CHESSR_log", "SNAPER", "ChEES"
```'''
algorithm_table = "\n".join(f"- `{a['id']}` (**{a['name']}**): {a['summary']}" for a in algorithms)
reference_list = "\n\n".join(f"{i}. {r['authors']} ({r['year']}). [{r['title']}]({r['url']}). {r['venue']}." for i, r in enumerate(references, 1))
template = (SOURCE / "README.template.md").read_text()
markdown = template.replace("{{QUICKSTART}}", quickstart).replace("{{ALGORITHMS}}", algorithm_table).replace("{{REFERENCES}}", reference_list)
## The reference-flow figure appears on the GitHub Pages site only; the README carries no figure.
markdown = re.sub(r"\{\{REFERENCE_FLOW\}\}\n+", "", markdown)
## The Nico logo sits beside the title in the GitHub README only (the website shows it in the banner).
markdown = markdown.replace("# NicoStan\n", '# NicoStan <img src="docs/assets/NicoStan_logo_720.png" align="right" height="140" alt="NicoStan logo" />\n', 1)
(ROOT / "README.md").write_text(markdown)


def slug(text):
    ## Headings that contain a link, e.g. "[BayesMVP](https://...): ...", are slugged from the link text only,
    ## matching GitHub's README anchors (e.g. #bayesmvp-multivariate-probit-models).
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9_ -]", "", text).replace(" ", "-")


def url(text):
    if text.startswith(("https://", "http://", "#")):
        return text
    if text.startswith("inst/examples/") and text.endswith((".stan", ".R")) and (ROOT / text).is_file():
        return EXAMPLE_PAGE + "#" + anchor_of_example_file(text)
    if text == "docs/references.bib":
        return "references.bib"
    return REPO + "/blob/main/" + text


def inline(text):
    result = []
    position = 0
    pattern = r"`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*"
    for match in re.finditer(pattern, text):
        result.append(escape(text[position:match.start()]))
        token = match.group()
        if token.startswith("`"):
            result.append("<code>" + escape(token[1:-1]) + "</code>")
        elif token.startswith("**"):
            ## Bold text is rendered recursively, so code and links inside it work ("**Specialised [BayesMVP](...) models:**"
            ## showed the raw link and its long URL pushed the page wider than a phone screen).
            result.append("<strong>" + inline(token[2:-2]) + "</strong>")
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            result.append('<a href="' + escape(url(link[2]), quote=True) + '">' + escape(link[1]) + "</a>")
        position = match.end()
    result.append(escape(text[position:]))
    return "".join(result)


svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 320" role="img" aria-labelledby="flow-title flow-desc">
<title id="flow-title">Unit Gaussian reference flow</title><desc id="flow-desc">Position u and velocity v rotate on a circle, preserving the reference energy. This shows the Gaussian substep only.</desc>
<rect width="480" height="320" rx="16" fill="#EEF0F5"/>
<g fill="none" stroke="#C9CDDB"><path d="M64 160H416M240 24V294"/><circle cx="240" cy="160" r="112" stroke-dasharray="3 5"/><circle cx="240" cy="160" r="72" opacity=".45"/></g>
<circle cx="240" cy="160" r="112" fill="none" stroke="#6272A4" stroke-width="2"/>
<path id="flow-radius" d="M240 160L322 236" stroke="#c16f45" stroke-width="2"/>
<circle id="flow-point" cx="322" cy="236" r="7" fill="#b25b35" stroke="#fff" stroke-width="3"/>
<g font-family="Georgia,serif" font-size="17" fill="#44475A"><text x="426" y="165">u</text><text x="234" y="19">v</text></g>
<g font-family="system-ui,sans-serif" font-size="12" fill="#6272A4"><text x="232" y="179">0</text><text x="349" y="178">1</text><text x="249" y="52">1</text><text x="22" y="301">GAUSSIAN REFERENCE SUBSTEP</text></g></svg>'''
(DOCS / "assets" / "reference-flow.svg").write_text(svg)
flow = '<figure id="reference-flow" class="flow"><div class="flow-plot">' + svg + '''</div><figcaption>
<span class="eyebrow">Explore the reference flow</span><h3>A Gaussian coordinate follows a rotation.</h3>
<p>Move the trajectory time to see position and velocity change. Their reference energy stays constant. The full coupled HMC proposal also contains numerical interaction steps.</p>
<label for="flow-time">Trajectory time <output id="flow-time-value">0.75</output></label>
<input id="flow-time" type="range" min="0" max="6.28" step="0.01" value="0.75"/>
<div class="flow-values"><span>u = <output id="flow-u">0.732</output></span><span>v = <output id="flow-v">-0.682</output></span><span>H₀ = <output id="flow-energy">0.500</output></span></div>
<p class="small">Initial state: u = 1, v = 0. Unit mass and unit Gaussian variance.</p></figcaption></figure>'''
tabs = '<div class="algorithm-explorer"><div class="algorithm-tabs" role="tablist" aria-label="Warm-up algorithm">'
for a in algorithms:
    tabs += f'<button role="tab" id="tab-{a["id"]}" aria-controls="panel-{a["id"]}" aria-selected="false" tabindex="-1">{escape(a["id"])}</button>'
tabs += "</div>"
for a in algorithms:
    tabs += f'<section class="algorithm-panel" id="panel-{a["id"]}" role="tabpanel" aria-labelledby="tab-{a["id"]}" tabindex="0"><h3>{escape(a["name"])}</h3><p>{escape(a["summary"])}</p><pre class="formula">{escape(a["formula"])}</pre><p class="small">{escape(a["note"])}</p></section>'
tabs += '</div><p class="small">The formulas show proposal acceptance weighting. Symbols and the log-ratio distinction are defined below.</p>'


def render(text):
    lines = text.splitlines()
    out, section_open = [], False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line in ("{{REFERENCE_FLOW}}", "{{ALGORITHMS}}"):
            out.append(flow if line == "{{REFERENCE_FLOW}}" else render(algorithm_table) + tabs)
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("## "):
            if section_open:
                out.append("</section>")
            title = line[3:]
            out.append(f'<section class="doc-section" id="{slug(title)}"><h2>{inline(title)}</h2>')
            section_open = True
            i += 1
            continue
        if line.startswith("### "):
            out.append(f'<h3 id="{slug(line[4:])}">{inline(line[4:])}</h3>')
            i += 1
            continue
        if line.startswith("```"):
            language = line[3:]
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            copy = '<button class="copy-code" type="button" aria-label="Copy R code">Copy</button>' if language == "r" else ""
            out.append(f'<div class="code-block">{copy}<pre><code class="language-{escape(language)}">{escape(chr(10).join(code))}</code></pre></div>')
            i += 1
            continue
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = lines[i].strip().strip("|").split("|")
                if not all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in row):
                    rows.append([inline(cell.strip()) for cell in row])
                i += 1
            out.append('<div class="table-scroll" role="region" aria-label="Model examples table" tabindex="0"><table><thead><tr>' + "".join("<th>"+c+"</th>" for c in rows[0]) + "</tr></thead><tbody>" + "".join("<tr>"+"".join("<td>"+c+"</td>" for c in row)+"</tr>" for row in rows[1:]) + "</tbody></table></div>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                item_lines = [lines[i][2:]]
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(("- ", "#", "```", "|", "{{")):
                    item_lines.append(lines[i].strip())
                    i += 1
                items.append("<li>" + inline(" ".join(item_lines)) + "</li>")
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"\d+\. ", line):
            out.append('<p class="reference">' + inline(line) + "</p>")
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "```", "|", "{{", "- ")):
            paragraph.append(lines[i]); i += 1
        out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    if section_open:
        out.append("</section>")
    return "\n".join(out)


template_for_site = re.sub(r"\$`([^`]+)`\$", r"$\1$", template)
body_text = template_for_site.replace("{{QUICKSTART}}", quickstart).replace("{{REFERENCES}}", reference_list)
## HTML comments are hidden on GitHub; drop them so the Pages site does not print them as text.
body_text = re.sub(r"<!--.*?-->[ \t]*\n?", "", body_text, flags=re.S)
body = render(body_text)
navigation = "".join(
    f'<a href="#{section_id}">{re.sub(r"<[^>]+>", "", title)}</a>'
    for section_id, title in re.findall(r'<section class="doc-section" id="([^"]+)"><h2>(.*?)</h2>', body)
)
## Banner title and paragraph come from website/website_banner.md (one website line per file line).
banner_lines = [line.rstrip("\n") for line in (SOURCE / "website_banner.md").read_text().split("\n") if not line.startswith("##")]
banner_description = next(line.split(":", 1)[1].strip() for line in banner_lines if line.startswith("description:"))
banner_lines = [line for line in banner_lines if not line.startswith("description:")]
banner_split = banner_lines.index("---")
banner_title_html = "<br>".join(escape(line) for line in banner_lines[:banner_split] if line.strip())
banner_intro_html = "<br>".join(escape(line) for line in banner_lines[banner_split + 1:] if line.strip())
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NicoStan | Adaptive HMC for Stan models</title><meta name="description" content="''' + escape(banner_description, quote=True) + '''">
<meta property="og:type" content="website"><meta property="og:site_name" content="NicoStan"><meta property="og:url" content="''' + SITE_URL + '''"><meta property="og:title" content="NicoStan | Adaptive HMC for Stan models"><meta property="og:description" content="''' + escape(banner_description, quote=True) + '''"><meta property="og:image" content="''' + SITE_URL + '''assets/NicoStan_social_card.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:image:alt" content="NicoStan logo"><meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#282A36"><link rel="icon" href="assets/NicoStan_logo_64.png" type="image/png"><link rel="stylesheet" href="assets/site.css"><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"><script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script><script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code'],throwOnError:false})"></script><script src="assets/site.js" defer></script><script data-goatcounter="https://cerulloe1996.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script></head><body>
<a class="skip-link" href="#content">Skip to content</a><header class="masthead"><a class="brand" href="#top"><img class="brand-logo" src="assets/NicoStan_logo_128.png" alt="" width="34" height="34">NicoStan</a><nav aria-label="Primary"><a href="#references">References</a><a href="#installation">Install</a><a href="https://github.com/CerulloE1996/NicoStan">GitHub ↗</a></nav></header>
<div class="hero" id="top"><div class="hero-copy"><p class="eyebrow">R package · Development version 0.1.9000</p><h1>''' + banner_title_html + '''</h1><p class="hero-intro">''' + banner_intro_html + '''</p><div class="hero-actions"><a class="button" href="#installation">Get started <span aria-hidden="true">↗</span></a></div><p class="byline">Developed by Enzo Cerullo</p></div><div class="hero-diagram"><img class="hero-logo" src="assets/NicoStan_logo_720.png" alt="NicoStan logo"></div></div>
<div class="layout"><aside><details open><summary>On this page</summary><nav aria-label="On this page">''' + navigation + '''</nav></details></aside><main id="content">''' + body + '''</main></div>
<footer><a class="brand" href="#top">NicoStan</a><p>Enzo Cerullo · GPL-3 · Development documentation</p><a href="references.bib">Download references (.bib)</a></footer></body></html>'''
if "\u2014" in page or "\u2014" in markdown:
    raise ValueError("An em dash was introduced into the generated prose")
(DOCS / "index.html").write_text(page)
(DOCS / ".nojekyll").write_text("")
## NicoStan_social_card.png (1200x630) is the link-preview image for Facebook etc. (see the og:image tag in the page head).
for name in ["site.css", "site.js", "NicoStan_logo_720.png", "NicoStan_logo_128.png", "NicoStan_logo_64.png", "NicoStan_social_card.png"]:
    shutil.copy2(SOURCE / name, DOCS / "assets" / name)
bibliography_source = SOURCE / "references.bib"
if bibliography_source.exists():
    shutil.copy2(bibliography_source, DOCS / "references.bib")
else:
    bibtex = []
    for r in references:
        # Use an explicit author list for correct BibTeX parsing of the short display citation.
        authors = re.sub(r", ([A-Z])", r", \1", r["authors"])
        bibtex.append("@misc{" + r["key"] + ",\n  title = {" + r["title"] + "},\n  author = {" + r.get("bibtex_authors", authors) + "},\n  year = {" + r["year"] + "},\n  howpublished = {" + r["venue"] + "},\n  url = {" + r["url"] + "}" + (",\n  doi = {" + r["doi"] + "}" if "doi" in r else "") + "\n}")
    (DOCS / "references.bib").write_text("\n\n".join(bibtex) + "\n")


## ---- The full code of every example: each Stan model (inst/examples, every folder) and each R script that runs
##      an example (Run_*.R, random_intercepts.R and the four-class scripts), in a collapsible panel with a Copy
##      button and a link to the file on GitHub; a link to a file's anchor opens its panel
## The benchmark models of Papers 1 and 3 (the current version of each, from the benchmark's models folder): each
## model's unchunked version (with Stan's fused functions), its chunked version and its chunked version with the
## hand-written AVX functions, where they exist; copied into inst/examples/benchmark_models (only when the copy is
## missing or older than the benchmark's file), so that the page and its GitHub links show the same files.
BENCHMARK_MODELS = Path.home() / ("Documents/Work/PhD_work/Alg_paper_analysis/paper_3_general_sampler/"
                                  "benchmark_criteria_diffusion_vs_standard_HMC_vs_Stan/models")
EXAMPLES_OF_THE_BENCHMARK = EXAMPLES / "benchmark_models"
benchmark_models = [
    ("Weibull survival", "Weibull_survival"),
    ("Shared frailty Cox survival", "shared_frailty_Cox_survival"),
    ("Stochastic volatility, non-centred", "stochastic_volatility_non_centred"),
    ("Stochastic volatility, centred", "stochastic_volatility_centred"),
    ("Latent diffusion survival", "latent_diffusion_survival"),
    ("Joint longitudinal and survival model", "joint_longitudinal_survival"),
    ("Two-compartment population pharmacokinetics", "two_compartment_population_pharmacokinetics"),
    ("Hierarchical logistic regression", "hierarchical_logistic_regression"),
    ("Robust Student-t regression", "robust_Student_t_regression"),
    ("Graded response IRT model", "graded_response_IRT"),
    ("Gaussian process regression", "Gaussian_process_regression"),
    ("Gaussian process classification", "Gaussian_process_classification"),
    ("Spatial logistic regression with a Hilbert space approximate Gaussian process",
     "spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process"),
    ("Four-class latent class multivariate ordinal probit (LC-MVOP)", "four_class_LC_MVOP"),
    ("Network meta-analysis of test accuracy (Xu et al., fixed cutpoints)",
     "network_meta_analysis_Xu_fixed_cutpoints"),
]
kinds_of_benchmark_file = [("unchunked, with Stan's fused functions",
                            "_unchunked_no_loops_over_N_with_Stan_fused_functions.stan"),
                           ("chunked", "_chunked.stan"),
                           ("chunked, with the hand-written AVX functions",
                            "_chunked_hand_written_BayesMVP_AVX.stan")]
# EXAMPLES_OF_THE_BENCHMARK.mkdir(exist_ok=True)
## (parents = True: the copy of the site that the sync script builds in a temporary folder has no inst/examples)
EXAMPLES_OF_THE_BENCHMARK.mkdir(parents = True, exist_ok = True)
benchmark_files_of_each_model = []
## The models whose current files do not follow the three names above: each file's kind, its source (in the
## benchmark's models folder) and the name of its copy
LIBRARIES_OF_THE_BENCHMARK = BENCHMARK_MODELS / "libraries_for_layout_timings"
COX_TIMED = ("shared_frailty_Cox_survival_one_fused_node_with_indexed_frailties_and_its_blocks_of_rows_"
             "run_in_parallel")
UNCHUNKED = "_unchunked_no_loops_over_N_with_Stan_fused_functions.stan"
files_of_the_models_with_other_names = {
    "stochastic_volatility_non_centred": [
        ("unchunked, with Stan's fused functions",
         BENCHMARK_MODELS / ("stochastic_volatility_non_centred" + UNCHUNKED)),
        ("chunked", BENCHMARK_MODELS / "stochastic_volatility_direct_sliced_boundary_scan_Stan_math.stan"),
        ("chunked, with the hand-written AVX functions",
         BENCHMARK_MODELS / "stochastic_volatility_direct_sliced_boundary_scan_hand_written_BayesMVP_AVX.stan")],
    "stochastic_volatility_centred": [
        ("unchunked, with Stan's fused functions",
         BENCHMARK_MODELS / ("stochastic_volatility_centred" + UNCHUNKED)),
        ("chunked", BENCHMARK_MODELS / "stochastic_volatility_centred_sliced_latent_path_Stan_math.stan"),
        ("chunked, with the hand-written AVX functions",
         BENCHMARK_MODELS / "stochastic_volatility_centred_sliced_latent_path_hand_written_BayesMVP_AVX.stan")],
    "shared_frailty_Cox_survival": [
        ("unchunked, with Stan's fused functions",
         BENCHMARK_MODELS / ("shared_frailty_Cox_survival" + UNCHUNKED)),
        ("chunked", LIBRARIES_OF_THE_BENCHMARK / (COX_TIMED + "_Stan_math_for_layout_timings.stan")),
        ("chunked, with the hand-written AVX functions",
         LIBRARIES_OF_THE_BENCHMARK / (COX_TIMED + "_hand_written_BayesMVP_AVX512_for_layout_timings.stan"))],
}
## The note shown under a model's heading
note_of_each_model = {
    "stochastic_volatility_non_centred": "NicoStan writes its nico_reduce_sum version itself, from "
                                         "nuisance_params.",
    "stochastic_volatility_centred": "NicoStan writes its nico_reduce_sum version itself, from "
                                     "nuisance_params.",
}
for title, model in benchmark_models:
    files = []
    if model in files_of_the_models_with_other_names:
        sources_of_this_model = files_of_the_models_with_other_names[model]
    else:
        sources_of_this_model = []
        for kind, ending in kinds_of_benchmark_file:
            source = BENCHMARK_MODELS / (model + ending)
            ## a model without an unchunked version with Stan's fused functions shows its own unchunked file
            if not source.is_file() and ending == kinds_of_benchmark_file[0][1]:
                source = BENCHMARK_MODELS / (model + ".stan")
            sources_of_this_model.append((kind, source))
    for kind, source in sources_of_this_model:
        if not source.is_file():
            continue
        copy = EXAMPLES_OF_THE_BENCHMARK / source.name.replace("_for_layout_timings", "")
        if not copy.is_file() or copy.stat().st_mtime < source.stat().st_mtime:
            shutil.copyfile(source, copy)
        files.append((kind, copy.relative_to(ROOT).as_posix()))
    if files:
        benchmark_files_of_each_model.append((title, files, note_of_each_model.get(model, "")))
## The other example models, each file shown once (two files with the same code under two names are shown once)
stan_files = []
codes_already_shown = set()
for path in sorted(EXAMPLES.rglob("*.stan")):
    if path.parent == EXAMPLES_OF_THE_BENCHMARK:
        continue
    code = path.read_text()
    if code in codes_already_shown:
        continue
    codes_already_shown.add(code)
    stan_files.append(path.relative_to(ROOT).as_posix())
R_files = sorted(path.relative_to(ROOT).as_posix() for path in EXAMPLES.rglob("*.R")
                 if path.name.startswith("Run_") or path.name in ("random_intercepts.R", "four_class_example.R",
                                                                  "run_four_class_example.R"))


def panel_of_example_file(relative_path, language):
    code = (ROOT / relative_path).read_text()
    n_lines = len(code.rstrip("\n").splitlines())
    return ('<details class="example-file" id="' + anchor_of_example_file(relative_path) + '"><summary><code>'
            + escape(relative_path.replace("inst/examples/", "")) + "</code> <span class=\"small\">(" + str(n_lines)
            + ' lines)</span></summary><div class="code-block"><button class="copy-code" type="button" '
            + 'aria-label="Copy the code">Copy</button><pre><code class="language-' + language + '">'
            + escape(code.rstrip("\n")) + '</code></pre></div><p class="small"><a href="' + REPO + "/blob/main/"
            + escape(relative_path, quote=True) + '">View on GitHub</a></p></details>')


def section_of_a_benchmark_model(title, files, note):
    return ("<h3>" + escape(title) + "</h3>" + ('<p class="small">' + escape(note) + "</p>" if note else "")
            + "".join('<p class="small">' + escape(kind) + "</p>"
                                                         + panel_of_example_file(f, "stan") for kind, f in files))


example_body = ('<h2 id="benchmark-models">Benchmark models</h2><p>The models of the benchmarks of NicoStan against '
                + "Stan, each in its current version: unchunked (with Stan's fused functions), chunked (tape chunking "
                + "with <code>reduce_sum</code>) and chunked with the hand-written AVX functions. Click a file to see "
                + "its full code.</p>"
                + "".join(section_of_a_benchmark_model(title, files, note)
                          for title, files, note in benchmark_files_of_each_model)
                + '<h2 id="stan-models">Other example models</h2><p>Each model, as NicoStan runs it in the examples '
                + "(the <code>_avx</code> files are the same models with the hand-written AVX functions).</p>"
                + "".join(panel_of_example_file(f, "stan") for f in stan_files)
                + '<h2 id="R-scripts">R scripts</h2><p>The scripts that run each example, from the data to the '
                + "summaries.</p>" + "".join(panel_of_example_file(f, "r") for f in R_files))
example_page = ('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NicoStan | Example models: full code</title><meta name="description" content="The full Stan code and R scripts of every NicoStan example model.">
<meta name="theme-color" content="#282A36"><link rel="icon" href="assets/NicoStan_logo_64.png" type="image/png"><link rel="stylesheet" href="assets/site.css">
<style>.example-file{border:1px solid var(--line);border-radius:7px;margin:0 0 12px;padding:12px 18px}.example-file summary{cursor:pointer;font-weight:550}.example-file[open] summary{margin-bottom:6px}</style>
<script src="assets/site.js" defer></script></head><body>
<a class="skip-link" href="#content">Skip to content</a><header class="masthead"><a class="brand" href="index.html"><img class="brand-logo" src="assets/NicoStan_logo_128.png" alt="" width="34" height="34">NicoStan</a><nav aria-label="Primary"><a href="index.html#examples">Examples</a><a href="index.html#installation">Install</a><a href="https://github.com/CerulloE1996/NicoStan">GitHub ↗</a></nav></header>
<div class="layout"><aside><details open><summary>On this page</summary><nav aria-label="On this page"><a href="#benchmark-models">Benchmark models</a><a href="#stan-models">Other example models</a><a href="#R-scripts">R scripts</a></nav></details></aside><main id="content"><h1>Example models: full code</h1>'''
                + example_body + '''</main></div>
<footer><a class="brand" href="index.html">NicoStan</a><p>Enzo Cerullo · GPL-3 · Development documentation</p></footer>
<script>function open_the_panel_of_the_link() { const target = document.getElementById(location.hash.slice(1));
if (target && target.tagName === "DETAILS") { target.open = true; target.scrollIntoView(); } }
window.addEventListener("hashchange", open_the_panel_of_the_link); open_the_panel_of_the_link();</script></body></html>''')
(DOCS / EXAMPLE_PAGE).write_text(example_page)
print("Built README.md, docs/index.html, docs/" + EXAMPLE_PAGE + " ("
      + str(sum(len(files) for _, files, _ in benchmark_files_of_each_model)) + " benchmark model files of "
      + str(len(benchmark_files_of_each_model)) + " models, " + str(len(stan_files)) + " other Stan models, "
      + str(len(R_files)) + " R scripts), static assets and the complete BibTeX references.")
