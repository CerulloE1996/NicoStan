"use strict";

const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectAlgorithm(tab) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectAlgorithm(tab));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault(); selectAlgorithm(tabs[next]); tabs[next].focus();
    }
  });
});
if (tabs.length) selectAlgorithm(tabs[0]);

const timeSlider = document.getElementById("flow-time");
function drawReferenceFlow() {
  const time = Number(timeSlider.value);
  const position = Math.cos(time), velocity = -Math.sin(time);
  const x = 240 + 112 * position, y = 160 - 112 * velocity;
  document.getElementById("flow-point").setAttribute("cx", String(x));
  document.getElementById("flow-point").setAttribute("cy", String(y));
  document.getElementById("flow-radius").setAttribute("d", `M240 160L${x} ${y}`);
  document.getElementById("flow-time-value").textContent = time.toFixed(2);
  document.getElementById("flow-u").textContent = position.toFixed(3);
  document.getElementById("flow-v").textContent = velocity.toFixed(3);
  document.getElementById("flow-energy").textContent = (0.5 * (position ** 2 + velocity ** 2)).toFixed(3);
}
if (timeSlider) { timeSlider.addEventListener("input", drawReferenceFlow); drawReferenceFlow(); }

document.querySelectorAll(".copy-code").forEach(button => {
  button.addEventListener("click", async () => {
    const text = button.parentElement.querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy"; }, 1800);
    } catch {
      button.textContent = "Select code to copy";
    }
  });
});

const sectionLinks = [...document.querySelectorAll('aside nav a')];
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of sectionLinks) {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "location"); else link.removeAttribute("aria-current");
      }
    }
  }, { rootMargin: "-5% 0px -72% 0px" });
  document.querySelectorAll(".doc-section").forEach(section => observer.observe(section));
}
