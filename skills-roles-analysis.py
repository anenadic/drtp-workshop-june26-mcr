import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# -----------------------------
# CONFIGURATION
# -----------------------------

# Google Sheet ID
# Replace with your own sheet ID
google_sheet_id = "12dhIpaEk2jpWvLNCBnGvuBEBQoiP1VISsbnSFMVHeG4"

# Optional: worksheet/tab name
sheet_name = "Reshaped"

role_columns = [
    "Data steward",
    "Data archivist",
    "Data librarian",
    "Data curator",
    "Data manager",
    "Data engineer",
    "Data scientist"
]

skill_column = "skill"

# -----------------------------
# LOAD + CLEAN DATA
# -----------------------------

# OLD LOCAL CSV VERSION
# csv_file = "skills-roles.csv"
# df = pd.read_csv(csv_file)

# NEW GOOGLE SHEETS VERSION
google_sheet_url = (
    f"https://docs.google.com/spreadsheets/d/"
    f"{google_sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
)

df = pd.read_csv(google_sheet_url)

# Clean column names
df.columns = df.columns.str.strip()

print("Raw/original data preview:")
print(df.head(10))

# -----------------------------
# SELECT DATA
# -----------------------------

skills = df[skill_column]

roles_df = df[role_columns]

# Convert to numeric
roles_df = roles_df.fillna(0)
roles_df = roles_df.apply(pd.to_numeric, errors='coerce')
roles_df = roles_df.round(0).astype(np.int64)

print("\nClean and rounded role data preview:")
print(roles_df.head(10))

# =========================================================
# PART 1 — SKILL PRESENCE (BINARY ANALYSIS)
# =========================================================

# Create binary dataframe
binary_df = (roles_df > 0).astype(int)
print("\nBinary data preview:")
print(binary_df.head(10))

roles = binary_df.columns

# Jaccard similarity matrix
jaccard_matrix = pd.DataFrame(
    index=roles,
    columns=roles
)

for r1 in roles:
    for r2 in roles:
        jaccard_matrix.loc[r1, r2] = jaccard_score(
            binary_df[r1],
            binary_df[r2]
        )

jaccard_matrix = jaccard_matrix.astype(float)

# -----------------------------
# VISUALISATION 1
# JACCARD HEATMAP
# -----------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    jaccard_matrix,
    annot=True,
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.title("Jaccard Similarity — Skill Presence")

plt.show()

# -----------------------------
# VISUALISATION 2
# DENDROGRAM
# -----------------------------

distance_matrix = pdist(
    binary_df.T,
    metric='jaccard'
)

linkage_matrix = linkage(
    distance_matrix,
    method='average'
)

plt.figure(figsize=(10, 6))

dendrogram(
    linkage_matrix,
    labels=roles,
    leaf_rotation=25
)

plt.title("Role Clustering — Skill Presence")

plt.show()

# =========================================================
# PART 2 — COMPETENCY LEVEL ANALYSIS
# =========================================================

# Cosine similarity
cosine_matrix = cosine_similarity(
    roles_df.T
)

cosine_df = pd.DataFrame(
    cosine_matrix,
    index=roles,
    columns=roles
)

# -----------------------------
# VISUALISATION 3
# COSINE HEATMAP
# -----------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    cosine_df,
    annot=True,
    cmap="Reds",
    vmin=0,
    vmax=1
)

plt.title("Cosine Similarity — Competency Levels")

plt.show()

# -----------------------------
# VISUALISATION 4
# CLUSTERED HEATMAP
# -----------------------------

# Remove constant columns for stability
cluster_df = roles_df.loc[
    :,
    roles_df.nunique() > 1
]

sns.clustermap(
    cluster_df.T,
    cmap="viridis",
    figsize=(14, 8)
)

# plt.show()

# ----------------------------------
# RADAR CHART
# ----------------------------------

# Use competency levels
radar_df = roles_df.fillna(0)

categories = skills.tolist()
N = len(categories)

angles = np.linspace(
    0,
    2 * np.pi,
    N,
    endpoint=False
).tolist()

angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(12, 12),
    subplot_kw=dict(polar=True)
)

for role in role_columns:

    values = radar_df[role].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=1,
        label=role
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)

ax.set_title(
    "Competency Profiles by Role",
    pad=30
)

ax.legend(
    bbox_to_anchor=(1.3, 1.1)
)

plt.show()