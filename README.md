# Sleep EEG App · CO3133

Deep Learning and Its Applications, Semester 261, HCMUT. Course project site for
Assignments 1–3 (instructor: Lê Thành Sách).

## Sleep EEG App

**An interactive EEG sleep-staging app, with all application source under
[`A2/webapp/`](A2/webapp/README.md).** Upload a Sleep-EDF recording, run the temporal
Transformer and AttnSleep, and review five-stage predictions alongside the waveform.

**Live demo: <https://quoc.eegsleep.dpdns.org/>**. Log in with username `admin` and
password `admin`. The demo is shared: every visitor sees the same recordings and can
upload, run and delete them, and uploads are removed automatically after 30 days. Use the
bundled sample or public Sleep-EDF files, never personal health data.

- Explore hypnograms, per-epoch probabilities and disagreements between the two models.
- Compare predictions with an optional expert hypnogram and export CSV/JSON results.
- **Find the top 5 similar EEG epochs** in an expert-scored reference library, using
  Transformer embeddings, AttnSleep embeddings or model-free signal features. Each
  result shows its waveform, expert stage, model predictions and similarity score.
- Inspect cross-space votes and training-exposure flags; recognized library recordings
  exclude their own participant from search. Model-based similarity reflects the
  encoder's learned representation and is evidence for review, not an independent verdict.

![Top-5 similar EEG epochs with expert labels, waveforms and three retrieval spaces](A2/webapp/docs/similar.png)

**Why find similar epochs, and why Qdrant.** In the library evaluation (60,462 epochs
unseen by the fold-5 models), Transformer and AttnSleep disagree on 8,330 epochs, and on
those neither is reliable: Transformer matches the expert stage 43% of the time, AttnSleep
40%. The two predictions alone cannot tell a researcher which model to trust, so the app
shows precedent instead. For the selected epoch it retrieves the 5 most similar of 195,469
expert-scored Sleep-EDF epochs and shows their waveforms and expert stages beside it. The
researcher compares the shapes, sees how experts scored epochs that look like this one, and
judges which model the evidence supports. A model's own feature space tends to agree with
that model: on the disagreements, the Transformer space sides with Transformer 65% of the
time and the AttnSleep space with AttnSleep 61%. The app therefore also offers a model-free
signal mode built from 12 EEG band powers, which sides with neither (40% vs 38%). When all
three modes give the same majority stage (69% of epochs), it matches the expert 91% of the
time. The library is stored in Qdrant, a vector database, as one point per epoch with three
named vectors: Transformer (128 dimensions) and AttnSleep (2,400 dimensions), both compared
by cosine similarity, and band powers (12 dimensions), compared by Euclidean distance.
Qdrant's HNSW index finds the top 5 without scanning every epoch and recovers 99.8–100% of
the exact nearest neighbours. Qdrant returns ids only; PostgreSQL holds each epoch's expert
stage and metadata, and the waveforms come from the library files.

**Deployment.** The application is one Docker Compose stack: an Nginx gateway (serves the
React build, routes `/api` to FastAPI, caps uploads and enforces Basic Auth), the FastAPI
API, a PyTorch inference worker that takes jobs from PostgreSQL, PostgreSQL itself and
Qdrant. The live demo runs this stack on an Ubuntu VPS with
[`compose.vps.yaml`](A2/webapp/compose.vps.yaml). No host ports are published: a Cloudflare
Tunnel provides HTTPS for the domain, the database and Qdrant stay on a private Docker
network, and every container has memory and CPU limits so the server can host other
projects alongside it. GitHub Actions runs the pytest suites and the Playwright end-to-end
tests against the full stack.

**Stack:** React / TypeScript / Vite, FastAPI / PyTorch, PostgreSQL, Qdrant, Nginx,
Docker Compose, Cloudflare Tunnel, pytest, Playwright and GitHub Actions.

[Live demo](https://quoc.eegsleep.dpdns.org/) ·
[App setup and library build](A2/webapp/README.md) ·
[VPS deployment guide](A2/webapp/docs/vps.md) ·
[EEG research and reproduction](A2/README.md) ·
[Mac demo guide](A2/webapp/docs/mac-demo.md)

The application uses trained checkpoints and a reference library, installed separately as
described in the app README. GitHub Pages serves the course pages; the application runs as
the Docker Compose stack above.

## Site structure

Assignment 1 setup, saved-result checks and reproduction commands are in
[A1/README.md](A1/README.md). See the [A1 overall workflow](A1/docs/overall.md)
for the shared data pipeline, training loop, and model guides.

Assignment 2's notebook-first full-waveform preparation, training, saved-run
metadata and evaluation instructions are in [A2/README.md](A2/README.md).
The browser application is in `A2/webapp/`, alongside the research pipeline.
Assignment 3 currently has a [multimodal project page](assignment3.html);
its `A3/` code directory has not been created yet.

```
index.html          shared landing page (institution, course, group, AI disclosure)
assignment1.html     Assignment 1 page
assignment2.html     Assignment 2 page
assignment3.html     Assignment 3 page
style.css            shared stylesheet
pics/                images for the site (logo)
AI_USAGE.md          detailed AI-tool usage log
A1/                  Assignment 1: Fashion-MNIST code, configs, runs, notebook and docs
A2/                  Assignment 2: EEG sleep staging research and application
├── webapp/          Sleep EEG App source and deployment setup
│   ├── frontend/   React/TypeScript UI and Playwright tests
│   ├── backend/    FastAPI, worker, ML inference, migrations and pytest tests
│   ├── docs/       Application screenshots, VPS and Mac demo guides
│   ├── compose.yaml
│   └── compose.vps.yaml  Server override used by the live demo
├── configs/         Research model configurations
├── sleep_exploration/  EDA and participant-fold manifests
├── runs/            Recorded research results and provenance
└── training.ipynb   Shared training/evaluation notebook
A3/                  Planned Assignment 3 code directory; not yet present
```

## Publish with GitHub Pages

1. Push these files to the root of your group's repository (`main` branch).
2. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   then pick branch `main`, folder `/ (root)`, and save.
3. The site will be live at `https://<username>.github.io/<repo-name>/` within a few minutes.
4. Replace every `[bracketed placeholder]` in the HTML files with real group info
   (member names, student IDs, roles, GitHub profiles, repo link) before
   the Week 3 skeleton deadline (09 Sep 2026, 23:59 GMT+7).

## Before each milestone deadline

- Fill in the relevant assignment page's sections (dataset, methodology, results, etc.)
  — the `class="todo"` boxes mark what's still missing.
- Update the AI usage disclosure on the landing page, the relevant assignment page,
  and `AI_USAGE.md`.
- Confirm the repo also contains, per assignment: config files, seed settings,
  dependency versions, hardware info, and checkpoint access/reconstruction instructions.
- Double-check every link on the page actually resolves (repo, checkpoints, report/slides,
  YouTube video) before submitting on the LMS.

## Reference dates (GMT+7, handbook Sec. 7)

| Date | Milestone |
|---|---|
| 09 Sep 2026 | Group + Pages skeleton (this deliverable) |
| 23 Sep 2026 | A1 M1 Draft |
| 07 Oct 2026 | A2 M1 Proposal |
| 21 Oct 2026 | A1 M2 Final |
| 28 Oct 2026 | A2 M2 Draft |
| 11 Nov 2026 | A2 M3 Final |
| 18 Nov 2026 | A3 M1 Proposal |
| 25 Nov 2026 | A3 M2 Draft |
| 02 Dec 2026 | A3 M3 Final |

The project handbook takes precedence over the teaching plan on exact weight
breakdowns; check the LMS for any official updates.
