import { getApiBase, getEngines, setApiBase } from "../api/client.js";
import { renderArenaPage } from "../arena/ArenaView.js";
import { createGamePage } from "./GamePage.js";
import "../styles/main.css";

const app = document.querySelector("#app");

async function boot() {
  const engines = (await getEngines()).engines;
  let activeTab = "play";
  let lastArenaResult = null;

  function render() {
    app.innerHTML = `
      <header class="topbar">
        <div>
          <strong>Chess960 Research Platform</strong>
          <span>Frontend client for the local learning backend</span>
        </div>
        <label class="api-base">API <input data-api-base value="${getApiBase()}" /></label>
      </header>
      <nav class="tabs">
        <button class="${activeTab === "play" ? "active" : ""}" data-tab="play">Play</button>
        <button class="${activeTab === "arena" ? "active" : ""}" data-tab="arena">Arena</button>
      </nav>
      <div data-page></div>
    `;
    app.querySelector("[data-api-base]").addEventListener("change", (event) => {
      setApiBase(event.target.value);
    });
    app.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        activeTab = button.dataset.tab;
        render();
      });
    });
    const page = app.querySelector("[data-page]");
    if (activeTab === "play") {
      page.appendChild(createGamePage({ engines }));
    } else {
      page.appendChild(
        renderArenaPage({
          engines,
          initialResult: lastArenaResult,
          onResult: (result) => (lastArenaResult = result),
        }),
      );
    }
  }

  render();
}

boot().catch((error) => {
  app.innerHTML = `<div class="fatal">Backend unavailable: ${error.message}</div>`;
});
