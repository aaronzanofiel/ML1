from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data():
    data_path = Path(__file__).resolve().parent / "placement_predict_50k Dataset (3)(in).csv"
    data = pd.read_csv(data_path)
    X = data[[
        "AttendancePercent",
        "CGPA"
    ]]
    return X


def elbow_method():
    X = load_data()
    K_value = range(2, 11)
    wcss = []

    for k in K_value:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    print("\nWCSS values:")
    for k, value in zip(K_value, wcss):
        print(f"K={k}: WCSS={value}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        K_value,
        wcss,
        marker="o",
        color="royalblue",
        linewidth=2,
        markersize=8
    )
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.xticks(K_value)
    plt.show()

    k = int(input("\nEnter K based on the Elbow method: "))
    return k


def silhouette_method():
    X = load_data()
    K_value = range(2, 11)
    silhouette_scores = []

    for k in K_value:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"Silhouette score for k={k}: {score:.4f}")

    best_index = silhouette_scores.index(max(silhouette_scores))
    best_k = list(K_value)[best_index]

    print(f"\nBest K value: {best_k}")
    print(f"Best Silhouette score: {silhouette_scores[best_index]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        K_value,
        silhouette_scores,
        marker="o",
        color="royalblue",
        linewidth=2,
        markersize=7
    )

    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Method")
    plt.grid(True)
    plt.xticks(K_value)
    plt.show()

    return best_k


if __name__ == "__main__":
    print("=== K-Means Clustering with Elbow and Silhouette Methods ===")

    # Run elbow method
    k_elbow = elbow_method()

    # Run silhouette method
    k_silhouette = silhouette_method()

    print(f"\n=== Results ===")
    print(f"K from Elbow Method: {k_elbow}")
    print(f"K from Silhouette Method: {k_silhouette}")