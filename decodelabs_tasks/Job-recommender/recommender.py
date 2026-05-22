import math
import csv


# ─────────────────────────────────────────────────────────────
# STEP 0 — Load Dataset  (raw_skills.csv)
# ─────────────────────────────────────────────────────────────

def load_dataset(filepath):
    """
    Load job roles and their skill tags from raw_skills.csv.
    Each row is one 'item' in our recommendation engine.
    Returns a list of dicts: [{job_role, skills: [...]}, ...]
    """
    dataset = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = row['skills'].strip().split()
            dataset.append({
                'job_role': row['job_role'].strip(),
                'skills':   skills
            })
    return dataset


# ─────────────────────────────────────────────────────────────
# STEP 1 — INGESTION  (User State Capture)
# ─────────────────────────────────────────────────────────────

def ingest_user_profile():
    """
    Capture at least 3 user skill inputs.
    Enforces minimum data density for accurate matching.
    Cold Start bypass: we force input so the user vector
    is never a zero-vector.
    """
    print("\n" + "="*54)
    print("   TECH STACK RECOMMENDER  |  DecodeLabs Project 3")
    print("="*54)
    print("\n[INGESTION] Enter your skills one by one.")
    print("  - Type a skill and press Enter")
    print("  - Minimum 3 skills required")
    print("  - Type 'done' when finished\n")

    user_skills = []
    while True:
        skill = input(f"  Skill {len(user_skills)+1}: ").strip()
        if skill.lower() == 'done':
            if len(user_skills) < 3:
                print(f"  [!] Need at least 3 skills. You have {len(user_skills)}. Keep going.\n")
            else:
                break
        elif skill:
            # Normalize: replace spaces with underscores (matches CSV format)
            skill = skill.replace(' ', '_')
            user_skills.append(skill)
            print(f"      Added: {skill}")

    print(f"\n  Profile captured: {user_skills}\n")
    return user_skills


# ─────────────────────────────────────────────────────────────
# VECTOR MAPPING  (Bridging the Language Barrier)
# ─────────────────────────────────────────────────────────────

def build_vocabulary(dataset):
    """
    Build a shared vocabulary space from all skill tags
    across every job role.

    As the PDF states:
      'Item features and user features must map to the
       exact same vocabulary.'

    Returns a sorted list of unique terms.
    """
    vocab = set()
    for role in dataset:
        for skill in role['skills']:
            vocab.add(skill.lower())
    return sorted(list(vocab))


# ─────────────────────────────────────────────────────────────
# TF-IDF WEIGHTING  (Upgrading Feature Extraction)
# ─────────────────────────────────────────────────────────────

def term_frequency(skill_list, vocab):
    """
    TF(t, d) = count(t in d) / total_terms(d)

    Terms appearing more in a document are more
    representative of it.
    """
    total = len(skill_list)
    counts = {}
    for s in skill_list:
        key = s.lower()
        counts[key] = counts.get(key, 0) + 1

    tf_vector = []
    for term in vocab:
        tf_vector.append(counts.get(term, 0) / total)

    return tf_vector


def inverse_document_frequency(vocab, dataset):
    """
    IDF(t) = log( N / df(t) )

    Penalizes terms that appear in many documents (generic).
    Rewards terms that appear in few documents (specific).

    The log creates a dampening effect so the penalty
    scales logarithmically, not linearly.
    """
    N = len(dataset)
    idf_vector = []

    for term in vocab:
        # Count how many roles contain this term
        df = sum(
            1 for role in dataset
            if any(s.lower() == term for s in role['skills'])
        )
        if df == 0:
            idf_vector.append(0.0)
        else:
            idf_vector.append(math.log(N / df))

    return idf_vector


def tfidf_vector(skill_list, vocab, idf_values):
    """
    TF-IDF(t, d) = TF(t, d) × IDF(t)

    Combines both scores. Unique, descriptive skills get
    high weight. Common, generic skills get penalized.
    """
    tf = term_frequency(skill_list, vocab)
    return [tf[i] * idf_values[i] for i in range(len(vocab))]


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY  (The Similarity Engine)
# ─────────────────────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """
    cos(θ) = (A · B) / (‖A‖ × ‖B‖)

    Measures the angle between two vectors.
    Invariant to magnitude — only orientation matters.

    Score = 1  →  perfectly aligned (identical interests)
    Score = 0  →  orthogonal (no shared characteristics)

    Why not Euclidean distance?
      Euclidean is sensitive to vector magnitude. A long
      job description vs a short one would score far apart
      even if they share the same tags. Cosine fixes this.
    """
    dot_product  = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a       = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b       = math.sqrt(sum(b ** 2 for b in vec_b))

    denominator  = norm_a * norm_b
    if denominator == 0:
        return 0.0  # Cold start guard: zero vector case

    return dot_product / denominator


# ─────────────────────────────────────────────────────────────
# THE 4-STEP RANKING PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(user_skills, dataset, top_n=3):
    """
    Implements the full 4-step pipeline from the PDF:

      1. Ingestion  — already done, user_skills is the input
      2. Scoring    — cosine similarity for every item
      3. Sorting    — descending by score
      4. Filtering  — Top-N list to prevent choice overload
    """

    # ── Build shared vocabulary & IDF values ──────────────────
    vocab      = build_vocabulary(dataset)
    idf_values = inverse_document_frequency(vocab, dataset)

    print("[PROCESS] Building TF-IDF vectors...")
    print(f"  Vocabulary size : {len(vocab)} unique terms")
    print(f"  Dataset size    : {len(dataset)} job roles\n")

    # ── User vector ───────────────────────────────────────────
    user_vec = tfidf_vector(user_skills, vocab, idf_values)

    # ── STEP 2: SCORING ──────────────────────────────────────
    # Loop through every item, compute cosine similarity
    scored_roles = []
    for role in dataset:
        role_vec = tfidf_vector(role['skills'], vocab, idf_values)
        score    = cosine_similarity(user_vec, role_vec)

        matched  = [s for s in role['skills']
                    if s.lower() in [u.lower() for u in user_skills]]

        scored_roles.append({
            'job_role':      role['job_role'],
            'skills':        role['skills'],
            'score':         score,
            'matched_skills': matched
        })

    # ── STEP 3: SORTING ──────────────────────────────────────
    # Descending order — highest similarity to the top
    scored_roles.sort(key=lambda x: x['score'], reverse=True)

    # ── STEP 4: FILTERING ─────────────────────────────────────
    # Truncate to Top-N to prevent choice overload
    top_results = scored_roles[:top_n]

    return top_results


# ─────────────────────────────────────────────────────────────
# OUTPUT  (Display Recommended Items)
# ─────────────────────────────────────────────────────────────

def display_results(results, user_skills):
    """
    Display the Top-N recommended career paths with
    similarity scores and matched skill highlights.
    """
    print("="*54)
    print("   OUTPUT — TOP RECOMMENDED CAREER PATHS")
    print("="*54)

    medals = ["🥇", "🥈", "🥉"]

    for i, role in enumerate(results):
        score_pct  = role['score'] * 100
        bar_filled = int(score_pct / 5)   # 20-char bar
        bar        = "█" * bar_filled + "░" * (20 - bar_filled)

        print(f"\n  {medals[i]}  #{i+1}  {role['job_role']}")
        print(f"      Similarity : [{bar}] {score_pct:.2f}%")
        print(f"      Matched    : {', '.join(role['matched_skills']) if role['matched_skills'] else 'None (TF-IDF weighted match)'}")
        print(f"      Role Skills: {', '.join(role['skills'])}")

    print("\n" + "="*54)
    print("  ALGORITHM SUMMARY")
    print("="*54)
    print(f"  Method       : Content-Based Filtering")
    print(f"  Vectorization: TF-IDF (Term Freq × Inverse Doc Freq)")
    print(f"  Similarity   : Cosine Similarity (angle-based)")
    print(f"  Your skills  : {', '.join(user_skills)}")
    print(f"  Top-N shown  : {len(results)} of 15 roles")
    print("="*54 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN — Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    # Load the dataset (raw_skills.csv — as referenced in PDF)
    dataset = load_dataset('raw_skills.csv')

    # STEP 1 — Ingestion: capture user profile (min 3 skills)
    user_skills = ingest_user_profile()

    # STEPS 2,3,4 — Scoring, Sorting, Filtering
    results = run_pipeline(user_skills, dataset, top_n=3)

    # Display recommended items
    display_results(results, user_skills)


if __name__ == '__main__':
    main()
