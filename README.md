# 🛍️ Customer Segmentation Using K-Means Cluster Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Transforming raw retail customer data into five actionable behavioral segments using K-Means clustering — enabling hyper-targeted marketing strategies and smarter business decisions.**

---

## 📌 Problem Statement

In a competitive retail environment, generic marketing campaigns waste budget and fail to resonate with diverse customer groups. A retail mall needed to move beyond one-size-fits-all promotions and instead understand the **behavioral and financial profiles** of its customers.

The challenge: given only customers' **annual income** and **spending scores**, can we discover meaningful, distinct groups — and build a marketing strategy around each?

This project applies unsupervised machine learning (K-Means clustering) to identify **5 distinct customer segments**, validate hypotheses about income-spending relationships, and deliver concrete, segment-specific business recommendations.

---

## 🎯 Business Objective

> Categorize retail mall customers into distinct behavioral groups to enable **personalized marketing**, improve **customer satisfaction**, and increase **profitability**.

### Hypotheses Tested

| Hypothesis | Result |
|------------|--------|
| Customers with higher annual income tend to have higher spending scores | ❌ Not universally true — two clusters show split high-income behavior |
| A group of low-income, high-spending customers exists | ✅ Confirmed — one distinct cluster exhibits this pattern |
| Middle-income customers display the most diverse spending patterns | ✅ Confirmed — shows both low and moderate spending |

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     ML PIPELINE                           │
│                                                           │
│  Raw Customer Dataset (CSV)                               │
│         │                                                 │
│         ▼                                                 │
│  Data Preprocessing                                       │
│  ├── Missing value check                                  │
│  ├── Outlier detection                                    │
│  └── StandardScaler normalization                         │
│         │                                                 │
│         ▼                                                 │
│  Exploratory Data Analysis (EDA)                          │
│  ├── Age / Income / Spending Score distributions          │
│  └── Correlation heatmap                                  │
│         │                                                 │
│         ▼                                                 │
│  Elbow Method ──► Optimal K = 5                           │
│         │                                                 │
│         ▼                                                 │
│  K-Means Clustering (k=5)                                 │
│         │                                                 │
│         ▼                                                 │
│  Cluster Visualization (2D + 3D)                          │
│         │                                                 │
│         ▼                                                 │
│  Business Insights & Recommendations                      │
└───────────────────────────────────────────────────────────┘
```

---

## 💡 Solution Approach

### 1. Data Preprocessing
- Verified dataset for null values and duplicates
- Selected features: **Annual Income (k$)** and **Spending Score (1–100)**
- Applied `StandardScaler` to normalize feature ranges and ensure equal weight in distance calculations

### 2. Exploratory Data Analysis
Key observations:
- **Age**: Concentrated between 20–40 years
- **Annual Income**: Right-skewed; most values clustered around $60k
- **Spending Score**: Peaks near 50 — average spenders dominate

### 3. Optimal Cluster Detection — Elbow Method
Plotted Within-Cluster Sum of Squares (WCSS) against `k` values (1–10).

> 📍 **Elbow identified at k = 5** — beyond this point, WCSS reduction becomes marginal, confirming 5 as the optimal cluster count.

### 4. K-Means Clustering
Applied `KMeans(n_clusters=5)` from scikit-learn. Generated both:
- **2D scatter plot** — Annual Income vs Spending Score with centroids
- **3D scatter plot** — Added Age as a third axis for enhanced segment separation

---

## 📊 Results & Cluster Profiles

### Customer Segments Identified

| Cluster | Label | Profile | Marketing Strategy |
|---------|-------|---------|-------------------|
| **1** | High Income, Low Spenders | Value-oriented, savings-focused | VIP perks, lifestyle campaigns, exclusive memberships |
| **2** | Low Income, High Spenders | Impulsive / brand-loyal shoppers | Bundled deals, loyalty discounts, BNPL options |
| **3** | Low Income, Low Spenders | Cost-conscious, budget-constrained | Clearance sales, price-match guarantees |
| **4** | High Income, High Spenders | Premium buyers — ideal target | Luxury loyalty programs, early product launches, exclusive events |
| **5** | Middle Income, Middle Spenders | Stable, reliable core customers | Seasonal promotions, steady engagement campaigns |

### Hypothesis Validation Summary
- Income alone does **not** determine spending — psychological and lifestyle factors play a significant role
- The low-income, high-spending cluster (Cluster 2) challenges conventional assumptions and demands dedicated marketing attention
- The middle-income group is **not** the most diverse as hypothesized, but shows moderate, predictable behavior

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core programming language |
| **Pandas** | Data loading, cleaning, and manipulation |
| **NumPy** | Numerical operations |
| **Scikit-learn** | K-Means algorithm, StandardScaler |
| **Matplotlib** | 2D and 3D cluster visualizations |
| **Seaborn** | Statistical distribution plots |

---

## 📁 Project Structure

```
customer-segmentation/
│
├── data/
│   └── mall_customers.csv              # Raw customer dataset
│
├── notebooks/
│   └── customer_segmentation.ipynb     # Full analysis: EDA → Elbow → KMeans
│
├── visuals/
│   ├── age_income_spending_dist.png    # EDA distributions
│   ├── elbow_method.png               # WCSS vs K plot
│   ├── clusters_2d.png                # 2D scatter with centroids
│   └── clusters_3d.png                # 3D scatter with Age axis
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Analysis
```bash
# Clone the repo
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation

# Launch Jupyter Notebook
jupyter notebook notebooks/customer_segmentation.ipynb
```

### Core Code Snippet
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Fit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

---

## ⚠️ Limitations

- Only **two features** used (Annual Income, Spending Score) — excludes behavioral variables like purchase frequency, product category, or visit duration
- K-Means assumes **spherical, equal-variance clusters** which may not reflect real-world complexity (non-convex shapes)
- Static snapshot analysis — does not account for **seasonal behavioral shifts**
- Dataset size may limit generalizability to the broader retail population

---

## 🌍 Project Impact

| Dimension | Impact |
|-----------|--------|
| **Revenue** | Targeted promotions for high-value Cluster 4 can significantly boost ROI |
| **Customer Retention** | Loyalty programs tailored per segment reduce churn |
| **Marketing Efficiency** | Reduces wasteful spend on non-converting customer groups |
| **Strategic Planning** | Segment evolution over time enables proactive inventory and staffing decisions |

---

## 🔮 Future Enhancements

- Add features: **Gender, Age, Visit Frequency, Product Categories**
- Explore advanced algorithms: **DBSCAN** (for non-spherical clusters), **Hierarchical Clustering** (for dendrogram insight)
- Implement **periodic re-clustering** to track behavioral drift over time
- Build an interactive dashboard (Power BI / Streamlit) for business stakeholder reporting

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.* Berkeley Symposium on Mathematical Statistics.
- Dataset: Mall Customer Segmentation Data — [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## 👤 Author

**[Mahima Srinivasan]**
- 📧 [mahima.s3994@gmail.com.com]
- 💼 [linkedin.com/in/mahimasrinivasan3994]

> *This project demonstrates proficiency in unsupervised machine learning, customer analytics, and translating data science outputs into actionable business strategy.*

---

*⭐ If you found this project helpful, please consider starring the repository!*
