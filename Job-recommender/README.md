# 🧠 Tech Stack Recommender
### AI-Powered Career Path Matching — DecodeLabs Project 3

A command-line recommendation engine that maps your skills to the most fitting tech career paths using **Content-Based Filtering**, **TF-IDF Weighting**, and **Cosine Similarity** — no neural networks, no black boxes, just clean math.

---

## 🚀 What It Does

You enter 3+ skills. The engine vectorizes them, computes similarity against 15 real-world job roles, and returns your top 3 career path matches — ranked by alignment score with a visual progress bar.

```
  🥇  #1  ML Engineer
      Similarity : [████████████████░░░░] 78.43%
      Matched    : Python, Machine_Learning, TensorFlow
      Role Skills: Python, Machine_Learning, TensorFlow, PyTorch, Deep_Learning ...

  🥈  #2  Data Scientist
      Similarity : [████████████░░░░░░░░] 61.20%
      ...
```

---

## 📐 How It Works — The Algorithm

This project implements a **4-Step Recommendation Pipeline** taught in the DecodeLabs AI/ML module:

```
  INPUT                PROCESS                   OUTPUT
  ─────                ───────                   ──────
  User Skills   →   [1] Ingestion            →   Ranked Career Paths
                    [2] TF-IDF Vectorization      + Similarity Scores
                    [3] Cosine Similarity          + Matched Skills
                    [4] Sort & Filter (Top-N)
```

### 1️⃣ Ingestion
Captures a minimum of 3 user-supplied skills. The minimum threshold is a **Cold Start bypass** — it ensures the user vector is never a zero-vector, preventing division-by-zero in similarity computation.

### 2️⃣ TF-IDF Vectorization

Each skill list (user's and each job role's) is converted to a numerical vector using TF-IDF weighting:

| Component | Formula | Purpose |
|---|---|---|
| **TF** | `count(t, d) / total_terms(d)` | How prominent is this skill in the profile? |
| **IDF** | `log(N / df(t))` | How rare/unique is this skill across all roles? |
| **TF-IDF** | `TF × IDF` | Combined weight — rewards rare, specific skills |

> Common skills like `Python` or `SQL` are penalized (high `df`). Niche skills like `MLOps` or `RTOS` are rewarded.

### 3️⃣ Cosine Similarity

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

Measures the **angular alignment** between the user vector and each job role vector. Unlike Euclidean distance, cosine similarity is **magnitude-invariant** — a short profile and a long one can still score perfectly if they share the same orientation.

- Score `1.0` → identical skill orientation
- Score `0.0` → completely orthogonal (no overlap)

### 4️⃣ Sort & Filter
Roles are sorted descending by similarity score. Only the Top-N (default: 3) are returned to prevent choice overload.

---

## 📁 Project Structure

```
tech-stack-recommender/
├── recommender.py      # Main engine — all logic lives here
├── raw_skills.csv      # Dataset — 15 job roles × ~10 skills each
└── README.md
```

---

## ▶️ Getting Started

**Requirements:** Python 3.7+ (no external libraries — uses only `math` and `csv`)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/tech-stack-recommender.git
cd tech-stack-recommender

# Run
python recommender.py
```

**Sample interaction:**

```
[INGESTION] Enter your skills one by one.

  Skill 1: Python
      Added: Python
  Skill 2: Machine Learning
      Added: Machine_Learning
  Skill 3: TensorFlow
      Added: TensorFlow
  Skill 4: done

  Profile captured: ['Python', 'Machine_Learning', 'TensorFlow']
```

---

## 📊 Dataset — `raw_skills.csv`

15 job roles, each tagged with ~10 relevant skills:

| # | Role | Sample Skills |
|---|------|--------------|
| 1 | Data Scientist | Python, ML, SQL, Statistics, TensorFlow |
| 2 | ML Engineer | Python, PyTorch, MLOps, Deep Learning |
| 3 | Backend Developer | Node.js, REST APIs, Docker, MongoDB |
| 4 | DevOps Engineer | Kubernetes, AWS, CI/CD, Terraform |
| 5 | Cybersecurity Analyst | Penetration Testing, SIEM, Cryptography |
| … | *(15 total)* | … |

---

## 🧪 Concepts Demonstrated

- **Content-Based Filtering** (not collaborative — no user-user data needed)
- **Vector Space Model** — skills as dimensions, profiles as coordinates
- **TF-IDF Weighting** — intelligent feature extraction
- **Cosine Similarity** — magnitude-invariant angular distance
- **Cold Start Problem** — handled via minimum input enforcement
- **IPO Architecture** — clean Input → Process → Output separation

---

## 📌 Notes

- Skills with spaces should be entered naturally (e.g., `Machine Learning`) — the engine normalizes them to `Machine_Learning` automatically to match the CSV format.
- To expand the dataset, add rows to `raw_skills.csv` following the existing format.
- `top_n` in `run_pipeline()` is configurable — change it to return more or fewer results.

---

## 👨‍💻 Author

👤 Name: Arslan Nafees<br>
   📱 Phone: +92 334 111 3047<br>
   📧 Email: arslannafees807@gmail.com<br>
[![GitHub](https://img.shields.io/badge/GitHub-arslannafees-181717?style=flat&logo=github)](https://github.com/arslannafees)
