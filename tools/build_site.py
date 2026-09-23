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
DOCS.mkdir(exist_ok=True)
(DOCS / "assets").mkdir(exist_ok=True)
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
## The reference-flow figure appears on the GitHub Pages site only; the README carries no figure (2026-09-22).
markdown = re.sub(r"\{\{REFERENCE_FLOW\}\}\n+", "", markdown)
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
            result.append("<strong>" + escape(token[2:-2]) + "</strong>")
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


body_text = template.replace("{{QUICKSTART}}", quickstart).replace("{{REFERENCES}}", reference_list)
## HTML comments are hidden on GitHub; drop them so the Pages site does not print them as text (2026-09-23).
body_text = re.sub(r"<!--.*?-->[ \t]*\n?", "", body_text, flags=re.S)
body = render(body_text)
navigation = "".join(
    f'<a href="#{section_id}">{re.sub(r"<[^>]+>", "", title)}</a>'
    for section_id, title in re.findall(r'<section class="doc-section" id="([^"]+)"><h2>(.*?)</h2>', body)
)
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NicoStan | Adaptive HMC for Stan models</title><meta name="description" content="Adaptive Hamiltonian Monte Carlo for general Stan models, with optional diffusion-pathspace HMC for Gaussian latent blocks. Methods, examples and installation.">
<meta name="theme-color" content="#282A36"><link rel="icon" href="assets/NicoStan_logo_64.png" type="image/png"><link rel="stylesheet" href="assets/site.css"><script src="assets/site.js" defer></script></head><body>
<a class="skip-link" href="#content">Skip to content</a><header class="masthead"><a class="brand" href="#top"><img class="brand-logo" src="assets/NicoStan_logo_128.png" alt="" width="34" height="34">NicoStan</a><nav aria-label="Primary"><a href="#references">References</a><a href="#installation">Install</a><a href="https://github.com/CerulloE1996/NicoStan">GitHub ↗</a></nav></header>
<div class="hero" id="top"><div class="hero-copy"><p class="eyebrow">R package · Development version 0.1.9000</p><h1>NicoStan:<br>Adaptive HMC<br>for Stan models.</h1><p class="hero-intro">NicoStan uses standard HMC for the main model parameters,<br>and diffusion-pathspace HMC for high-dimensional nuisance parameters/Gaussian latent variables.<br>NicoStan also uses adaptive between-chain adaptation to achieve rapid burnin/warmup, <br>and state-of-the-art burnin algorithms with a variety of options, such as ChESSR-HMC and SNAPER-HMC.<br>NicoStan also offers rapid parallel estimation of posterior summaries, and only monitors and stores the trace for the main model parameters (by default).<br>NicoStan also works for general Stan models without nuisance parameters.</p><div class="hero-actions"><a class="button" href="#installation">Get started <span aria-hidden="true">↗</span></a></div><p class="byline">Developed by Enzo Cerullo</p></div><div class="hero-diagram"><img class="hero-logo" src="assets/NicoStan_logo_720.png" alt="NicoStan logo"></div></div>
<div class="layout"><aside><details open><summary>On this page</summary><nav aria-label="On this page">''' + navigation + '''</nav></details></aside><main id="content">''' + body + '''</main></div>
<footer><a class="brand" href="#top">NicoStan</a><p>Enzo Cerullo · GPL-3 · Development documentation</p><a href="references.bib">Download references (.bib)</a></footer></body></html>'''
if "\u2014" in page or "\u2014" in markdown:
    raise ValueError("An em dash was introduced into the generated prose")
(DOCS / "index.html").write_text(page)
(DOCS / ".nojekyll").write_text("")
for name in ["site.css", "site.js", "NicoStan_logo_720.png", "NicoStan_logo_128.png", "NicoStan_logo_64.png"]:
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
print("Built README.md, docs/index.html, static assets and the complete BibTeX references.")
