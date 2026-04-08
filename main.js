const svgPlaceholder = (title, subtitle, start = "#0b2039", end = "#0d2f52") => {
  const safeTitle = title.replace(/&/g, "&amp;");
  const safeSubtitle = subtitle.replace(/&/g, "&amp;");
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${start}"/>
        <stop offset="100%" stop-color="${end}"/>
      </linearGradient>
      <radialGradient id="r" cx="15%" cy="20%" r="80%">
        <stop offset="0%" stop-color="rgba(0,229,255,0.25)"/>
        <stop offset="100%" stop-color="rgba(0,229,255,0)"/>
      </radialGradient>
    </defs>
    <rect width="1280" height="720" fill="url(#g)"/>
    <rect width="1280" height="720" fill="url(#r)"/>
    <g opacity="0.35">
      <circle cx="180" cy="110" r="6" fill="#00e5ff"/>
      <circle cx="1120" cy="610" r="4" fill="#6cffb8"/>
      <circle cx="960" cy="160" r="3" fill="#ff6b4a"/>
    </g>
    <text x="70" y="365" fill="#edf7ff" font-size="72" font-family="Space Grotesk, sans-serif" font-weight="700">${safeTitle}</text>
    <text x="72" y="420" fill="#95b7d6" font-size="28" font-family="Manrope, sans-serif">${safeSubtitle}</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg.trim())}`;
};

const CONTENT = {
  project: {
    title: "100Editor: 100+ Views per Batch and\nMinute-Scale View-Consistent 3D Editing",
    subtitle: "CVPR 2026(findings)",
    abstract:
      "Editing 3D scenes with diffusion models and 3DGS remains slow: most pipelines update one view at a time and are constrained by VRAM-limited batch sizes. We introduce 100Editor, a training-free framework that scales multi-view 3D editing to the hundred-view regime while preserving cross-view consistency. The system combines four complementary components: (1) a batch-consistent multi-view editing module that aligns overlapping content across views at the token level with chunked execution; (2) an efficiency suite that integrates a lightweight 3DGS renderer, a sparse optimizer, a CLIP-guided Patience Stopping (CPS) rule, and parallelized diffusion inference to reduce editing latency; (3) an interactive 3D segmentation module with point prompts and 3D back-projection for accurate, object-level local editing; and (4) a practical 3D editing software that unifies these capabilities for semantic, additive, subtractive, and non-rigid editing. Together, these designs enable large-batch and efficient 3D editing without modifying the image-editing model's weights. On a single 24 GB GPU, 100Editor edits 100+ views per batch (up to 120) and achieves minute-scale 3D scene editing latency (59.60~s). Experiments across diverse scenes and edit types show improved multi-view consistency and high perceptual quality compared to single-view and small-batch baselines, while supporting precise, interactive 3D edits. ",
    authorInfo: {
      authors: [
        { name: "Cunqi Wu", superscript: "1" },
        { name: "Peng Zhou", superscript: "†,1" },
        { name: "Jie Qin", superscript: "1" },
        { name: "Qi Tian", superscript: "2" },
      ],
      contribution: "† Corresponding author.",
      affiliations: [
        "1 Nanjing University of Aeronautics and Astronautics",
        "2 Huawei Inc.",
      ],
    },
  },
  hero: {
    video: "assets/teaser_merged_loop.gif",
    poster: svgPlaceholder("", "", "#d9edf8", "#c3ebe3"),
  },
  metrics: [
    { label: "PSNR Gain vs Baselines", value: 4.8, suffix: " dB" },
    { label: "Inference Speed", value: 31, suffix: " FPS" },
    { label: "Error Reduction", value: 37, suffix: "%" },
    { label: "Cross-Domain Robustness", value: 2.4, suffix: "x" },
  ],
  highlights: [
    {
      title: "Cross-Scale Semantic Fusion",
      desc: "Couples global context with local reconstruction to recover texture under severe corruption.",
      icon: "S",
    },
    {
      title: "Temporal Stability Engine",
      desc: "Suppresses frame jitter and preserves continuity for dynamic scenes without ghosting artifacts.",
      icon: "T",
    },
    {
      title: "Fast Lightweight Decoder",
      desc: "Deployment-ready architecture that keeps quality high while hitting real-time throughput.",
      icon: "F",
    },
  ],
  methodViews: [
    {
      key: "pipeline",
      image: "assets/overall.png",
      alt: "100Editor Pipeline",
      captionLead: "Overview of 100Editor.",
      captionBody:
        "100Editor primarily comprises three key modules: (1) a large-batch, view-consistent editing module that enhances multi-view consistency during batch editing via view token merging; (2) a fast editing module that accelerates both 3DGS optimization and diffusion inference, enabling minute-scale editing latency; and (3) an interactive 3D segmentation module that leverages point prompts and 3D back-projection to enable precise editing.",
    },
    {
      key: "parallel",
      image: "assets/Parallel.png",
      alt: "Encoder-Cached Parallel Decoding",
      captionLead: "Encoder-Cached Parallel Decoding.",
      captionBody: "We enable parallel decoding for multiple non-key steps to accelerate diffusion inference.",
    },
  ],
  cases: [
    {
      id: "face",
      name: "face",
      icon: "fas fa-user",
      viewerSource: "./models/face.html",
      viewerEdited: "./models/vampire.html",
      inputMedia: svgPlaceholder("Input", "Noisy / low-light", "#091325", "#151a2f"),
      outputMedia: svgPlaceholder("Output", "Enhanced details + color", "#072233", "#10495f"),
      note: "Case 01: Turn his face into a vampire.",
    },
    {
      id: "plant",
      name: "plant",
      icon: "fas fa-leaf",
      viewerSource: "./models/plant.html",
      viewerEdited: "./models/rose.html",
      inputMedia: svgPlaceholder("Input", "Motion-corrupted frame", "#0e1528", "#261631"),
      outputMedia: svgPlaceholder("Output", "Clear structure recovery", "#0a2536", "#235272"),
      note: "Case 02: There is a rose on the branch.",
    },
  ],
  results: {
    semantic: {
      title: "Semantic Editing",
      kind: "video-pair-grid",
      cards: [
        {
          prompt: "put makeup on her",
          sourceVideo: "./assets/results/girl.mp4",
          editedVideo: "./assets/results/make_up.mp4",
        },
        {
          prompt: "Turn it into a tiger",
          sourceVideo: "./assets/results/kanagroo.mp4",
          editedVideo: "./assets/results/tiger.mp4",
        },
        {
          prompt: "Turn his face into a Harry Potter",
          sourceVideo: "./assets/results/face.mp4",
          editedVideo: "./assets/results/Harry_Potter.mp4",
        },
        {
          prompt: "Turn him into a Dead Pool",
          sourceVideo: "./assets/results/yuseung.mp4",
          editedVideo: "./assets/results/Dead_pool.mp4",
        },
        {
          prompt: "Make it winter",
          sourceVideo: "./assets/results/garden.mp4",
          editedVideo: "./assets/results/winter.mp4",
        },
        {
          prompt: "Turn the bear into a corgi",
          sourceVideo: "./assets/results/bear.mp4",
          editedVideo: "./assets/results/corgi.mp4",
        },
        {
          prompt: "Turn the orange into a red apple",
          sourceVideo: "./assets/results/orange.mp4",
          editedVideo: "./assets/results/apple.mp4",
        },
        {
          prompt: "Turn the truck into red",
          sourceVideo: "./assets/results/truck.mp4",
          editedVideo: "./assets/results/red_truck.mp4",
        },
        {
          prompt: "Make it fire",
          sourceVideo: "./assets/results/stump.mp4",
          editedVideo: "./assets/results/fire.mp4",
        },
      ],
    },
    additive: {
      title: "Additive Editing",
      kind: "video-pair-grid",
      cards: [
        {
          prompt: "A man wears a black mask",
          sourceVideo: "./assets/results/add_face.mp4",
          editedVideo: "./assets/results/black_mask.mp4",
        },
        {
          prompt: "It wears Martens boots, wings, and metal ears.",
          sourceVideo: "./assets/results/add_kangroo.mp4",
          editedVideo: "./assets/results/kangroo_wind.mp4",
        },
        {
          prompt: "A toy wears a Santas hat",
          sourceVideo: "./assets/results/dtu_scan.mp4",
          editedVideo: "./assets/results/santas.mp4",
        },
      ]
    },
    subtractive: {
      title: "Subtractive Editing",
      kind: "video-pair-grid",
      cards: [
        {
          prompt: "delete the mouse",
          sourceVideo: "./assets/results/mouse.mp4",
          editedVideo: "./assets/results/delete_mouse.mp4",
        },
        {
          prompt: "delete the plant",
          sourceVideo: "./assets/results/counter.mp4",
          editedVideo: "./assets/results/delete_counter.mp4",
        },
        {
          prompt: "delete the case",
          sourceVideo: "./assets/results/garden_delete.mp4",
          editedVideo: "./assets/results/delete_garden.mp4",
        },
      ]
    },
    nonrigid: {
      title: "Non-rigid Editing",
      kind: "video-pair-grid",
      cards: [
        {
          prompt: "Raise your arms",
          sourceVideo: "./assets/results/person.mp4",
          editedVideo: "./assets/results/person_put_his_hands.mp4",
        },
      ]
    },
  },
  quantitative: {
    rows: [
      { method: "GaussianEditor", clipSim: 0.2222, clipDir: 0.1039, userStudy: 12.34, time: 184.29 },
      { method: "GaussCtrl", clipSim: 0.2389, clipDir: 0.1209, userStudy: 9.42, time: 259.86 },
      { method: "DGE", clipSim: 0.2455, clipDir: 0.1525, userStudy: 20.78, time: 133.95 },
      { method: "EditSplat", clipSim: 0.2471, clipDir: 0.1318, userStudy: 15.58, time: 470.44 },
      { method: "100Editor", clipSim: 0.2597, clipDir: 0.1726, userStudy: 41.88, time: 59.6, isOurs: true },
    ],
  },
  links: {
    paper: "https://arxiv.org/abs/0000.00000",
    arxiv: "https://arxiv.org/abs/0000.00000",
    code: "https://github.com/Devin100086/100Editor",
    bilibili: "https://www.bilibili.com",
    software: "https://github.com/Devin100086/100Editor",
    viewer: "https://github.com/Devin100086/Nebula",
  },
};

const BIBTEX = `@inproceedings{photonweaver2026,
  title={PhotonWeaver: Seeing Through Extreme Conditions},
  author={Doe, Jane and Zhang, Alex and Team, Visionary Intelligence},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}`;

let activeCaseId = "";
let activeSceneMode = "source";
let hasCountedMetrics = false;
let toasting = null;
let reducedParallax = false;
const THEME_STORAGE_KEY = "mvp-theme";
const ABSTRACT_HIGHLIGHT_SENTENCE =
  "On a single 24 GB GPU, 100Editor edits 100+ views per batch (up to 120) and achieves minute-scale 3D scene editing latency (59.60~s).";
const RESULTS_CARDS_PER_PAGE = 9;

const byId = (id) => document.getElementById(id);

const QUANT_COLUMNS = [
  {
    key: "method",
    label: "Method",
    type: "text",
    better: "higher",
  },
  {
    key: "clipSim",
    label: "CLIPsim",
    labelHtml: "CLIP<sub>sim</sub>",
    type: "number",
    better: "higher",
  },
  {
    key: "clipDir",
    label: "CLIPdir",
    labelHtml: "CLIP<sub>dir</sub>",
    type: "number",
    better: "higher",
  },
  {
    key: "userStudy",
    label: "User Study",
    type: "number",
    better: "higher",
  },
  {
    key: "time",
    label: "Time",
    type: "number",
    better: "lower",
  },
];

function formatQuantValue(columnKey, value) {
  if (columnKey === "userStudy") return `${value.toFixed(2)}%`;
  if (columnKey === "time") return `${value.toFixed(2)}s`;
  return value.toFixed(4);
}

function cleanMethodLabel(methodName) {
  if (typeof methodName !== "string") return "";
  return methodName
    .replace(/\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\u3010\s*\d+\s*\u3011|\uff08\s*\d+\s*\uff09)+\s*$/gu, "")
    .trim();
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderAbstract(abstractEl, abstractText) {
  const safeAbstract = escapeHtml(abstractText);
  const safeHighlight = escapeHtml(ABSTRACT_HIGHLIGHT_SENTENCE);

  if (safeAbstract.includes(safeHighlight)) {
    abstractEl.innerHTML = safeAbstract.replace(
      safeHighlight,
      `<span class="abstract-highlight">${safeHighlight}</span>`
    );
    return;
  }

  abstractEl.textContent = abstractText;
}

const isMobile = () => window.matchMedia("(max-width: 900px)").matches;
const NAV_COLLAPSE_QUERY = "(max-width: 1200px)";

function applyTheme(theme) {
  const nextTheme = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", nextTheme);

  const toggleButton = byId("theme-toggle");
  if (!toggleButton) return;

  const darkMode = nextTheme === "dark";
  toggleButton.textContent = darkMode ? "🌙" : "☀";
  toggleButton.setAttribute("aria-label", darkMode ? "Switch to light mode" : "Switch to dark mode");
  toggleButton.setAttribute("title", darkMode ? "Switch to light mode" : "Switch to dark mode");
  toggleButton.setAttribute("aria-pressed", String(!darkMode));
}

function initThemeToggle() {
  const toggleButton = byId("theme-toggle");
  if (!toggleButton) return;

  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  } catch (error) {
    console.warn("Cannot read saved theme preference", error);
  }
  applyTheme(savedTheme || "light");

  toggleButton.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch (error) {
      console.warn("Cannot save theme preference", error);
    }
  });
}

function initResponsiveNav() {
  const navToggle = byId("nav-toggle");
  const nav = byId("primary-nav");
  if (!navToggle || !nav) return;

  const mq = window.matchMedia(NAV_COLLAPSE_QUERY);
  const isCollapsedViewport = () => mq.matches;

  const setMenuState = (expanded) => {
    const isOpen = isCollapsedViewport() ? expanded : false;
    document.body.classList.toggle("nav-open", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
    navToggle.setAttribute("aria-label", isOpen ? "Close navigation menu" : "Open navigation menu");
    navToggle.setAttribute("title", isOpen ? "Close navigation menu" : "Open navigation menu");
    nav.setAttribute("aria-hidden", String(isCollapsedViewport() ? !isOpen : false));
  };

  const closeMenu = () => setMenuState(false);

  navToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!isCollapsedViewport()) return;
    const expanded = navToggle.getAttribute("aria-expanded") === "true";
    setMenuState(!expanded);
  });

  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (!target.closest("a[href]")) return;
    closeMenu();
  });

  document.addEventListener("click", (event) => {
    if (!isCollapsedViewport()) return;
    if (navToggle.getAttribute("aria-expanded") !== "true") return;
    const target = event.target;
    if (nav.contains(target) || navToggle.contains(target)) return;
    closeMenu();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    closeMenu();
  });

  const syncViewportState = () => closeMenu();
  if (typeof mq.addEventListener === "function") {
    mq.addEventListener("change", syncViewportState);
  } else {
    mq.addListener(syncViewportState);
  }

  closeMenu();
}

function initProjectContent() {
  byId("project-subtitle").textContent = CONTENT.project.subtitle;
  const abstractEl = byId("project-abstract");
  if (CONTENT.project.abstract) {
    renderAbstract(abstractEl, CONTENT.project.abstract);
    abstractEl.classList.remove("hidden");
  } else {
    abstractEl.classList.add("hidden");
  }
  const taglineEl = byId("project-tagline");
  if (CONTENT.project.tagline) {
    taglineEl.textContent = CONTENT.project.tagline;
    taglineEl.classList.remove("hidden");
  } else {
    taglineEl.classList.add("hidden");
  }
  byId("hero-title").textContent = CONTENT.project.title;
  renderProjectAuthors(CONTENT.project.authorInfo);
  animateHeroTitle();
  byId("bibtex-block").textContent = BIBTEX;
}

function renderProjectAuthors(authorInfo) {
  const authorsRoot = byId("project-authors");
  if (!authorsRoot) return;
  authorsRoot.textContent = "";

  if (!authorInfo || !Array.isArray(authorInfo.authors) || authorInfo.authors.length === 0) {
    authorsRoot.classList.add("hidden");
    return;
  }

  const listEl = document.createElement("p");
  listEl.className = "hero-author-list";
  authorInfo.authors.forEach((author, index) => {
    const nameEl = author.url ? document.createElement("a") : document.createElement("span");
    nameEl.className = "hero-author-name";
    nameEl.textContent = author.name;
    if (author.url) {
      nameEl.href = author.url;
      nameEl.target = "_blank";
      nameEl.rel = "noreferrer noopener";
    }
    listEl.appendChild(nameEl);

    if (author.superscript) {
      const supEl = document.createElement("sup");
      supEl.className = "hero-author-sup";
      supEl.textContent = author.superscript;
      listEl.appendChild(supEl);
    }

    if (index < authorInfo.authors.length - 1) {
      listEl.appendChild(document.createTextNode(", "));
    }
  });
  authorsRoot.appendChild(listEl);

  if (authorInfo.contribution) {
    const contributionEl = document.createElement("p");
    contributionEl.className = "hero-author-note";
    contributionEl.textContent = authorInfo.contribution;
    authorsRoot.appendChild(contributionEl);
  }

  if (Array.isArray(authorInfo.affiliations) && authorInfo.affiliations.length > 0) {
    const affiliationsWrap = document.createElement("div");
    affiliationsWrap.className = "hero-author-affiliations";
    authorInfo.affiliations.forEach((affiliation) => {
      const lineEl = document.createElement("p");
      lineEl.className = "hero-author-affiliation";
      const match = affiliation.match(/^(\d+)\s*(.*)$/);
      if (match) {
        const markerEl = document.createElement("sup");
        markerEl.textContent = match[1];
        lineEl.append(markerEl, ` ${match[2]}`);
      } else {
        lineEl.textContent = affiliation;
      }
      affiliationsWrap.appendChild(lineEl);
    });
    authorsRoot.appendChild(affiliationsWrap);
  }

  authorsRoot.classList.remove("hidden");
}

function initHeroMedia() {
  const heroSection = byId("hero");
  const heroVideo = byId("hero-video");
  const links = CONTENT.links || {};
  heroVideo.poster = CONTENT.hero.poster || "";

  let heroImage = byId("hero-image");
  if (!heroImage) {
    heroImage = document.createElement("img");
    heroImage.id = "hero-image";
    heroImage.className = "hero-video hidden";
    heroImage.alt = "Hero background";
    heroImage.loading = "eager";
    heroImage.decoding = "async";
    heroImage.setAttribute("aria-hidden", "true");
    heroSection.insertBefore(heroImage, heroVideo.nextSibling);
  }

  const showImageMedia = (source) => {
    heroVideo.pause();
    heroVideo.removeAttribute("src");
    heroVideo.load();
    heroVideo.classList.add("hidden");
    heroImage.src = encodeURI(source);
    heroImage.classList.remove("hidden");
    heroSection.classList.remove("is-fallback");
  };

  const showVideoMedia = () => {
    heroImage.classList.add("hidden");
    heroVideo.classList.remove("hidden");
  };

  const isImageSource = (source) => /\.(gif|png|jpe?g|webp|avif)(?:[?#].*)?$/i.test(source || "");

  const playlist = Array.isArray(CONTENT.hero.videos) && CONTENT.hero.videos.length
    ? CONTENT.hero.videos
    : CONTENT.hero.video
      ? [CONTENT.hero.video]
      : [];

  if (playlist.length > 0) {
    if (playlist.length === 1) {
      const source = playlist[0];
      if (isImageSource(source)) {
        showImageMedia(source);
      } else {
        showVideoMedia();
        heroVideo.loop = true;
        heroVideo.muted = true;
        heroVideo.playsInline = true;
        heroVideo.preload = "auto";
        heroVideo.src = encodeURI(source);
        heroVideo.load();
        heroVideo.play().catch(() => {
          // Autoplay may be blocked on some browsers; keep muted/inline config as fallback.
        });
        heroVideo.addEventListener("error", () => heroSection.classList.add("is-fallback"), { once: true });
      }
    } else {
      showVideoMedia();
      const setVideoSource = (videoEl, source) => {
        if (videoEl.dataset.source === source) return;
        videoEl.dataset.source = source;
        videoEl.src = encodeURI(source);
        videoEl.load();
      };

      const playSilently = (videoEl) => {
        videoEl.play().catch(() => {
          // Autoplay may be blocked on some browsers; keep muted/inline config as fallback.
        });
      };

      // Keep two players: one visible and one preloaded for the next segment.
      const bufferedVideo = heroVideo.cloneNode(false);
      bufferedVideo.id = "hero-video-buffer";
      bufferedVideo.removeAttribute("loop");
      bufferedVideo.style.opacity = "0";
      bufferedVideo.style.transition = "opacity 180ms linear";
      heroVideo.style.transition = "opacity 180ms linear";
      heroSection.insertBefore(bufferedVideo, heroVideo.nextSibling);

      const players = [heroVideo, bufferedVideo];
      let activePlayerIndex = 0;
      let activePlaylistIndex = 0;

      players.forEach((player, idx) => {
        player.loop = false;
        player.muted = true;
        player.playsInline = true;
        player.preload = "auto";
        player.style.opacity = idx === 0 ? "0.62" : "0";
        player.addEventListener("error", () => heroSection.classList.add("is-fallback"), { once: true });
      });

      setVideoSource(players[0], playlist[0]);
      if (playlist.length > 1) {
        setVideoSource(players[1], playlist[1]);
      }
      playSilently(players[0]);

      const handleEnded = (endedPlayerIndex) => {
        if (endedPlayerIndex !== activePlayerIndex) return;

        const incomingPlayerIndex = 1 - activePlayerIndex;
        const outgoingPlayer = players[activePlayerIndex];
        const incomingPlayer = players[incomingPlayerIndex];

        activePlaylistIndex = (activePlaylistIndex + 1) % playlist.length;

        const revealIncoming = () => {
          incomingPlayer.removeEventListener("loadeddata", revealIncoming);
          incomingPlayer.currentTime = 0;
          playSilently(incomingPlayer);
          incomingPlayer.style.opacity = "0.62";
          outgoingPlayer.style.opacity = "0";
          outgoingPlayer.pause();
        };

        if (incomingPlayer.readyState >= 2) {
          revealIncoming();
        } else {
          incomingPlayer.addEventListener("loadeddata", revealIncoming, { once: true });
        }

        activePlayerIndex = incomingPlayerIndex;

        const nextPlaylistIndex = (activePlaylistIndex + 1) % playlist.length;
        setVideoSource(outgoingPlayer, playlist[nextPlaylistIndex]);
        outgoingPlayer.currentTime = 0;
      };

      players[0].addEventListener("ended", () => handleEnded(0));
      players[1].addEventListener("ended", () => handleEnded(1));
    }
  } else {
    heroSection.classList.add("is-fallback");
  }

  const bindHeroLink = (id, href) => {
    const linkEl = byId(id);
    if (!linkEl) return;
    if (!href) {
      linkEl.classList.add("hidden");
      return;
    }
    linkEl.classList.remove("hidden");
    linkEl.href = href;
    if (/^(https?:)?\/\//.test(href)) {
      linkEl.target = "_blank";
      linkEl.rel = "noreferrer noopener";
    } else {
      linkEl.removeAttribute("target");
      linkEl.removeAttribute("rel");
    }
  };

  bindHeroLink("hero-paper-link", links.paper);
  bindHeroLink("hero-arxiv-link", links.arxiv);
  bindHeroLink("hero-code-link", links.code);
  bindHeroLink("hero-bilibili-link", links.bilibili);
  bindHeroLink("hero-software-link", links.software);
  bindHeroLink("hero-viewer-link", links.viewer);
}

function animateHeroTitle() {
  const titleEl = byId("hero-title");
  const text = titleEl.textContent;
  const lines = text.split("\n");

  titleEl.innerHTML = "";
  titleEl.setAttribute("aria-label", lines.join(" ").trim());

  let charIndex = 0;
  lines.forEach((line) => {
    const lineEl = document.createElement("span");
    lineEl.className = "hero-title-line";

    [...line].forEach((char) => {
      const span = document.createElement("span");
      span.className = "char";
      span.style.setProperty("--char-index", String(charIndex));
      span.textContent = char === " " ? "\u00a0" : char;
      lineEl.appendChild(span);
      charIndex += 1;
    });

    titleEl.appendChild(lineEl);
  });
}

function formatMetricValue(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function animateMetricNumber(element, target, suffix = "") {
  const duration = 1300;
  const start = performance.now();

  const tick = (now) => {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const next = target * eased;
    element.textContent = `${formatMetricValue(next)}${suffix}`;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function buildMetricCard(metric) {
  const card = document.createElement("article");
  card.className = "metric-card";
  const value = document.createElement("span");
  value.className = "metric-value";
  value.dataset.target = String(metric.value);
  value.dataset.suffix = metric.suffix || "";
  value.textContent = "0";
  const label = document.createElement("p");
  label.className = "metric-label";
  label.textContent = metric.label;
  card.append(value, label);
  return card;
}

function renderMetrics() {
  const track = byId("impact-track");
  if (!track) return;
  track.textContent = "";
  const cards = CONTENT.metrics.map((metric) => buildMetricCard(metric));
  const clone = CONTENT.metrics.map((metric) => buildMetricCard(metric));
  [...cards, ...clone].forEach((card) => track.appendChild(card));
}

function renderHighlights() {
  const grid = byId("highlights-grid");
  if (!grid) return;
  grid.textContent = "";
  CONTENT.highlights.forEach((item) => {
    const card = document.createElement("article");
    card.className = "highlight-card";
    card.innerHTML = `
      <div class="highlight-icon" aria-hidden="true">${item.icon}</div>
      <h3>${item.title}</h3>
      <p>${item.desc}</p>
    `;
    grid.appendChild(card);
  });
}

function renderMethodShowcase() {
  const methodImage = byId("method-image");
  const methodSwitch = byId("method-switch");
  const methodCaption = byId("method-caption");
  if (!methodImage || !methodSwitch) return;

  const methodByKey = new Map((CONTENT.methodViews || []).map((item) => [item.key, item]));
  const buttons = [...methodSwitch.querySelectorAll(".method-tab")];
  if (buttons.length === 0 || methodByKey.size === 0) return;

  const applyMethodView = (key) => {
    const target = methodByKey.get(key);
    if (!target) return;

    methodImage.src = target.image;
    methodImage.alt = target.alt;
    if (methodCaption) {
      methodCaption.textContent = "";
      const lead = document.createElement("strong");
      lead.className = "method-caption-lead";
      lead.textContent = target.captionLead || target.alt || "";
      methodCaption.appendChild(lead);

      if (target.captionBody) {
        methodCaption.appendChild(document.createTextNode(" "));
        const body = document.createElement("span");
        body.className = "method-caption-body";
        body.textContent = target.captionBody;
        methodCaption.appendChild(body);
      }
    }
    buttons.forEach((button) => {
      const selected = button.dataset.methodView === key;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => applyMethodView(button.dataset.methodView || ""));
  });

  const defaultKey = buttons.find((button) => button.classList.contains("is-active"))?.dataset.methodView;
  applyMethodView(defaultKey || buttons[0].dataset.methodView || "");
}

function preloadImage(source) {
  return new Promise((resolve) => {
    if (!source) {
      resolve();
      return;
    }
    const image = new Image();
    image.src = source;
    image.onload = () => resolve();
    image.onerror = () => resolve();
  });
}

function renderCases() {
  const tabs = byId("case-tabs");
  tabs.textContent = "";
  CONTENT.cases.forEach((item, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = `tab-${item.id}`;
    btn.className = "case-tab";
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-controls", "scene-panel");
    btn.setAttribute("aria-selected", String(index === 0));
    btn.setAttribute("aria-label", item.name);

    if (item.icon) {
      const icon = document.createElement("i");
      icon.className = `${item.icon} case-tab-icon`;
      icon.setAttribute("aria-hidden", "true");
      btn.appendChild(icon);
    }

    const label = document.createElement("span");
    label.className = "case-tab-label";
    label.textContent = item.name;
    btn.appendChild(label);

    btn.addEventListener("click", () => switchCase(item.id));
    tabs.appendChild(btn);
  });

  if (window.FontAwesome?.dom?.i2svg) {
    window.FontAwesome.dom.i2svg({ node: tabs });
  }

  activeCaseId = CONTENT.cases[0]?.id || "";
  if (activeCaseId) switchCase(activeCaseId, true);
}

function updateSceneToggleState() {
  const sourceLabel = byId("scene-source-label");
  const editedLabel = byId("scene-edited-label");
  const editedMode = activeSceneMode === "edited";
  sourceLabel?.classList.toggle("is-active", !editedMode);
  editedLabel?.classList.toggle("is-active", editedMode);
}

function initSceneToggle() {
  const sceneToggle = byId("scene-toggle");
  if (!sceneToggle) return;

  activeSceneMode = sceneToggle.checked ? "edited" : "source";
  updateSceneToggleState();

  sceneToggle.addEventListener("change", () => {
    activeSceneMode = sceneToggle.checked ? "edited" : "source";
    updateSceneToggleState();
    if (activeCaseId) {
      switchCase(activeCaseId, false, true);
    }
  });
}

function setTabSelection(caseId) {
  const buttons = document.querySelectorAll(".case-tab");
  buttons.forEach((button) => {
    const selected = button.id === `tab-${caseId}`;
    button.setAttribute("aria-selected", String(selected));
  });
}

function syncSceneMedia(selectedCase, showEdited) {
  const sceneImage = byId("demo-media");
  const viewerShell = byId("demo-viewer-shell");
  const viewerFrame = byId("demo-viewer-frame");
  if (!sceneImage || !viewerShell || !viewerFrame) return;

  const viewerSource = showEdited ? selectedCase.viewerEdited : selectedCase.viewerSource;
  const showViewer = Boolean(viewerSource);

  if (showViewer) {
    if (viewerFrame.dataset.src !== viewerSource) {
      viewerFrame.dataset.src = viewerSource;
      viewerFrame.src = viewerSource;
    }
    sceneImage.classList.add("hidden");
    viewerShell.classList.remove("hidden");
    return;
  }

  viewerShell.classList.add("hidden");
  sceneImage.classList.remove("hidden");
}

async function switchCase(caseId, instant = false, forceRefresh = false) {
  if (!caseId || (caseId === activeCaseId && !instant && !forceRefresh)) return;
  const selectedCase = CONTENT.cases.find((item) => item.id === caseId);
  if (!selectedCase) return;

  const sceneImage = byId("demo-media");
  const sceneFrame = byId("scene-panel").querySelector(".media-frame");

  setTabSelection(caseId);
  byId("demo-note").textContent = selectedCase.note;
  activeCaseId = caseId;

  await Promise.all([preloadImage(selectedCase.inputMedia), preloadImage(selectedCase.outputMedia)]);

  if (!instant) {
    sceneFrame.classList.add("is-swapping");
  }

  const setAssets = () => {
    const showEdited = activeSceneMode === "edited";
    sceneImage.src = showEdited ? selectedCase.outputMedia : selectedCase.inputMedia;
    sceneImage.alt = `${selectedCase.name} ${showEdited ? "edited scene" : "source scene"}`;
    syncSceneMedia(selectedCase, showEdited);
  };

  if (instant) {
    setAssets();
    return;
  }

  setTimeout(setAssets, 125);
  setTimeout(() => {
    sceneFrame.classList.remove("is-swapping");
  }, 270);
}

function renderResultsMode(modeId) {
  const panel = byId("results-panel");
  if (!panel) return;

  const mode = CONTENT.results?.[modeId];
  if (!mode) {
    panel.textContent = "";
    return;
  }

  if (mode.kind === "video-pair-grid") {
    const cards = Array.isArray(mode.cards) ? mode.cards : [];
    const totalPages = Math.max(1, Math.ceil(cards.length / RESULTS_CARDS_PER_PAGE));
    const savedPage = Number(panel.dataset.page || "1");
    const currentPage = Math.min(Math.max(savedPage, 1), totalPages);
    const start = (currentPage - 1) * RESULTS_CARDS_PER_PAGE;
    const visibleCards = cards.slice(start, start + RESULTS_CARDS_PER_PAGE);

    const cardsHtml = visibleCards
      .map(
        (card, index) => `
      <article class="result-card">
        <p class="result-prompt">
          <img
            class="result-prompt-icon"
            src="assets/wand.png"
            alt=""
            aria-hidden="true"
            loading="lazy"
            decoding="async"
          />
          <span class="result-prompt-text">${card.prompt || ""}</span>
        </p>
        <div class="result-video-pair">
          <div class="result-video-block">
            <video
              src="${card.sourceVideo || ""}"
              autoplay
              muted
              loop
              playsinline
              controls
              preload="metadata"
              aria-label="${mode.title} card ${start + index + 1} original video"
            ></video>
          </div>
          <div class="result-video-divider" aria-hidden="true"></div>
          <div class="result-video-block">
            <video
              src="${card.editedVideo || ""}"
              autoplay
              muted
              loop
              playsinline
              controls
              preload="metadata"
              aria-label="${mode.title} card ${start + index + 1} edited video"
            ></video>
          </div>
        </div>
      </article>
    `
      )
      .join("");

    const paginationHtml =
      totalPages > 1
        ? `
      <nav class="results-pagination" aria-label="Results pages">
        ${Array.from({ length: totalPages }, (_, i) => {
          const page = i + 1;
          const activeClass = page === currentPage ? "is-active" : "";
          const ariaCurrent = page === currentPage ? 'aria-current="page"' : "";
          return `<button type="button" class="results-page-btn ${activeClass}" data-results-page="${page}" ${ariaCurrent}>${page}</button>`;
        }).join("")}
      </nav>
    `
        : "";

    panel.dataset.page = String(currentPage);
    panel.innerHTML = `<div class="results-page-grid">${cardsHtml}</div>${paginationHtml}`;

    if (totalPages > 1) {
      panel.querySelectorAll(".results-page-btn").forEach((button) => {
        button.addEventListener("click", () => {
          const nextPage = Number(button.dataset.resultsPage || "1");
          if (!Number.isFinite(nextPage) || nextPage === currentPage) return;
          panel.dataset.page = String(nextPage);
          renderResultsMode(modeId);
        });
      });
    }
    return;
  }

  panel.innerHTML = `
    <article class="result-card result-card-placeholder">
      <p class="result-prompt">
        <img
          class="result-prompt-icon"
          src="assets/wand.png"
          alt=""
          aria-hidden="true"
          loading="lazy"
          decoding="async"
        />
        <span class="result-prompt-text">${mode.title}</span>
      </p>
      <p class="result-placeholder-text">Results for this mode will be added soon.</p>
    </article>
  `;
}

function initResultsGallery() {
  const panel = byId("results-panel");
  const buttons = [...document.querySelectorAll(".result-mode-btn")];
  if (!panel || buttons.length === 0) return;

  const activateMode = (modeId) => {
    buttons.forEach((button) => {
      const isActive = button.dataset.resultMode === modeId;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
    const previousMode = panel.dataset.mode || "";
    if (previousMode !== modeId) {
      panel.dataset.page = "1";
    }
    panel.dataset.mode = modeId;
    renderResultsMode(modeId);
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const nextMode = button.dataset.resultMode || "semantic";
      activateMode(nextMode);
    });
  });

  const defaultMode = buttons.find((button) => button.classList.contains("is-active"))?.dataset.resultMode || "semantic";
  activateMode(defaultMode);
}

function renderQuantitativeTable() {
  const root = byId("quantitative-root");
  if (!root) return;

  const rows = CONTENT.quantitative?.rows;
  if (!Array.isArray(rows) || rows.length === 0) {
    root.textContent = "";
    return;
  }

  const numericColumns = QUANT_COLUMNS.filter((column) => column.type === "number");
  const columnByKey = new Map(QUANT_COLUMNS.map((column) => [column.key, column]));
  const originalOrder = new Map(rows.map((row, index) => [row.method, index]));

  const metricStats = {};
  numericColumns.forEach((column) => {
    const values = rows.map((row) => row[column.key]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const best = column.better === "lower" ? min : max;
    metricStats[column.key] = { min, max, best };
  });

  const scoreForValue = (column, value) => {
    const stats = metricStats[column.key];
    if (!stats || stats.max === stats.min) return 1;
    const normalized = (value - stats.min) / (stats.max - stats.min);
    return column.better === "lower" ? 1 - normalized : normalized;
  };

  const isBestValue = (column, value) => {
    const stats = metricStats[column.key];
    if (!stats) return false;
    return Math.abs(value - stats.best) < 1e-9;
  };

  let sortState = { key: null, direction: "desc" };

  root.textContent = "";

  const card = document.createElement("div");
  card.className = "quant-card";

  const toolbar = document.createElement("div");
  toolbar.className = "quant-toolbar";

  const hint = document.createElement("p");
  hint.id = "quant-hint";
  hint.className = "quant-hint";
  hint.textContent = "Click a column header to sort. Amber cells indicate best values.";

  const sortStatus = document.createElement("p");
  sortStatus.className = "quant-sort-status";
  sortStatus.textContent = "Default order";

  toolbar.append(hint, sortStatus);
  card.appendChild(toolbar);

  const tableWrap = document.createElement("div");
  tableWrap.className = "quant-table-wrap";

  const table = document.createElement("table");
  table.className = "quant-table";
  table.setAttribute("aria-describedby", "quant-hint");

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const headerRefs = [];

  QUANT_COLUMNS.forEach((column) => {
    const th = document.createElement("th");
    th.scope = "col";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "quant-sort-btn";
    button.dataset.key = column.key;
    button.setAttribute("aria-label", `Sort by ${column.label}`);
    button.setAttribute("aria-pressed", "false");

    const label = document.createElement("span");
    label.className = "quant-sort-label";
    if (column.labelHtml) {
      label.innerHTML = column.labelHtml;
    } else {
      label.textContent = column.label;
    }

    const icon = document.createElement("span");
    icon.className = "quant-sort-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "↕";

    button.append(label, icon);
    button.addEventListener("click", () => {
      if (sortState.key === column.key) {
        sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
      } else {
        sortState.key = column.key;
        sortState.direction = column.type === "number" && column.better === "lower" ? "asc" : "desc";
      }
      renderRows();
    });

    th.appendChild(button);
    headRow.appendChild(th);
    headerRefs.push({ column, th, button, icon });
  });

  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  table.append(thead, tbody);
  tableWrap.appendChild(table);
  card.appendChild(tableWrap);
  root.appendChild(card);

  const getSortedRows = () => {
    const sorted = [...rows];
    if (!sortState.key) return sorted;

    const column = columnByKey.get(sortState.key);
    if (!column) return sorted;

    sorted.sort((a, b) => {
      let result = 0;
      if (column.type === "number") {
        result = a[column.key] - b[column.key];
      } else {
        result = a[column.key].localeCompare(b[column.key], "en", { sensitivity: "base" });
      }

      if (result === 0) {
        result = (originalOrder.get(a.method) || 0) - (originalOrder.get(b.method) || 0);
      }

      return sortState.direction === "asc" ? result : -result;
    });

    return sorted;
  };

  const renderRows = () => {
    const orderedRows = getSortedRows();
    tbody.textContent = "";

    orderedRows.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.isOurs) tr.classList.add("is-ours");

      const methodCell = document.createElement("th");
      methodCell.scope = "row";
      methodCell.className = "quant-method-cell";

      const methodName = document.createElement("span");
      methodName.className = "quant-method-name";
      methodName.textContent = cleanMethodLabel(row.method);

      if (row.isOurs) {
        const championIcon = document.createElement("span");
        championIcon.className = "quant-champion-icon";
        championIcon.textContent = "🏆";
        championIcon.setAttribute("role", "img");
        championIcon.setAttribute("aria-label", "Champion");
        championIcon.title = "Champion";
        methodCell.appendChild(championIcon);

        methodCell.appendChild(methodName);

        const badge = document.createElement("span");
        badge.className = "quant-ours-badge";
        badge.textContent = "Ours";
        methodCell.appendChild(badge);
      } else {
        methodCell.appendChild(methodName);
      }

      tr.appendChild(methodCell);

      numericColumns.forEach((column) => {
        const value = row[column.key];
        const td = document.createElement("td");
        td.className = "quant-metric-cell";

        if (isBestValue(column, value)) {
          td.classList.add("is-best");
        }

        const metricValue = document.createElement("span");
        metricValue.className = "quant-metric-value";
        metricValue.textContent = formatQuantValue(column.key, value);

        td.appendChild(metricValue);
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    headerRefs.forEach(({ column, th, button, icon }) => {
      const active = sortState.key === column.key;
      const direction = active ? sortState.direction : null;
      th.setAttribute("aria-sort", direction === "asc" ? "ascending" : direction === "desc" ? "descending" : "none");
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      icon.textContent = active ? (direction === "asc" ? "↑" : "↓") : "↕";
    });

    if (!sortState.key) {
      sortStatus.textContent = "Default order";
      return;
    }

    const activeColumn = columnByKey.get(sortState.key);
    const directionLabel = sortState.direction === "asc" ? "ascending" : "descending";
    sortStatus.textContent = `Sorted by ${activeColumn?.label || "metric"} (${directionLabel})`;
  };

  renderRows();
}

function initSoftwarePills() {
  const group = document.querySelector(".software-actions");
  if (!group) return;

  const pills = [...group.querySelectorAll(".software-pill")];
  if (pills.length === 0) return;
  const softwareVideo = byId("software-video");

  const updateSoftwareVideo = (activePill) => {
    if (!softwareVideo || !activePill) return;
    const nextSource = activePill.dataset.softwareVideo;
    if (!nextSource) return;

    softwareVideo.muted = true;
    softwareVideo.loop = true;
    softwareVideo.playsInline = true;

    if (softwareVideo.dataset.source !== nextSource) {
      softwareVideo.dataset.source = nextSource;
      softwareVideo.src = nextSource;
      softwareVideo.load();
    }

    softwareVideo.play().catch(() => {
      // Autoplay may be blocked in some browsers despite muted inline setup.
    });
  };

  const setActivePill = (activePill) => {
    pills.forEach((pill) => {
      const selected = pill === activePill;
      pill.classList.toggle("is-active", selected);
      pill.setAttribute("aria-pressed", String(selected));
    });
    updateSoftwareVideo(activePill);
  };

  const defaultPill = pills.find((pill) => pill.classList.contains("is-active")) || pills[0];
  setActivePill(defaultPill);

  pills.forEach((pill) => {
    pill.addEventListener("click", () => setActivePill(pill));
  });
}

function renderLinkButtons() {
  const linkRoot = byId("link-buttons");
  linkRoot.textContent = "";
  const items = [];

  items.forEach((item) => {
    if (!item.href) return;
    const link = document.createElement("a");
    link.className = "btn btn-ghost";
    link.href = item.href;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = item.label;
    linkRoot.appendChild(link);
  });
}

function runMetricCounter() {
  if (hasCountedMetrics) return;
  hasCountedMetrics = true;
  document.querySelectorAll(".metric-value").forEach((el) => {
    const target = Number(el.dataset.target || 0);
    const suffix = el.dataset.suffix || "";
    animateMetricNumber(el, target, suffix);
  });
}

function bindScrollAnimations() {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );
  document.querySelectorAll(".reveal").forEach((item) => revealObserver.observe(item));

  const metricObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          runMetricCounter();
          metricObserver.disconnect();
        }
      });
    },
    { threshold: 0.25 }
  );
  metricObserver.observe(byId("impact"));
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(toasting);
  toasting = setTimeout(() => toast.classList.remove("is-visible"), 1400);
}

async function copyBibtex() {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(BIBTEX);
      showToast("BibTeX copied");
      return;
    }
  } catch (error) {
    console.warn("Clipboard API failed, trying fallback", error);
  }

  const temp = document.createElement("textarea");
  temp.value = BIBTEX;
  temp.setAttribute("readonly", "");
  temp.style.position = "absolute";
  temp.style.left = "-9999px";
  document.body.appendChild(temp);
  temp.select();
  document.execCommand("copy");
  temp.remove();
  showToast("BibTeX copied");
}

function initParticleField() {
  const canvas = byId("particle-canvas");
  const context = canvas.getContext("2d");
  if (!context) return;

  const maxParticles = isMobile() ? 40 : 92;
  const particles = [];
  let width = 0;
  let height = 0;
  let raf = 0;

  const resize = () => {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * devicePixelRatio);
    canvas.height = Math.floor(height * devicePixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  };

  const seedParticles = () => {
    particles.length = 0;
    for (let i = 0; i < maxParticles; i += 1) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.36,
        vy: (Math.random() - 0.5) * 0.36,
        r: Math.random() * 1.8 + 0.5,
      });
    }
  };

  const draw = () => {
    context.clearRect(0, 0, width, height);
    for (let i = 0; i < particles.length; i += 1) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -20) p.x = width + 20;
      if (p.x > width + 20) p.x = -20;
      if (p.y < -20) p.y = height + 20;
      if (p.y > height + 20) p.y = -20;

      context.beginPath();
      context.fillStyle = "rgba(0, 229, 255, 0.55)";
      context.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      context.fill();
    }

    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d > 120) continue;
        const alpha = (1 - d / 120) * 0.18;
        context.strokeStyle = `rgba(108, 255, 184, ${alpha.toFixed(3)})`;
        context.lineWidth = 0.7;
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
    }

    raf = requestAnimationFrame(draw);
  };

  resize();
  seedParticles();
  draw();

  window.addEventListener("resize", () => {
    cancelAnimationFrame(raf);
    resize();
    seedParticles();
    draw();
  });
}

function configurePerformanceMode() {
  reducedParallax = isMobile();
  if (reducedParallax) {
    document.body.classList.add("mobile-mode");
  }
}

function initEvents() {
  byId("copy-bibtex").addEventListener("click", copyBibtex);
}

function bootstrap() {
  initThemeToggle();
  initResponsiveNav();
  configurePerformanceMode();
  initProjectContent();
  initHeroMedia();
  renderMetrics();
  renderMethodShowcase();
  initSoftwarePills();
  initSceneToggle();
  renderCases();
  initResultsGallery();
  renderQuantitativeTable();
  renderLinkButtons();
  bindScrollAnimations();
  initParticleField();
  initEvents();
}

window.addEventListener("DOMContentLoaded", bootstrap);
