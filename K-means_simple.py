import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(
    "placement_predict_50k Dataset (3)(in).csv"
)
X=data[[
    "AttendancePercent",
    "CGPA"
]
]

print("\nSelected features ")
print(X.head)

wcss=[]

for k in range(1,11):
    kmeans=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(
    range(1, 11),
    wcss,
    marker="o",
    color="royalblue",
    linewidth=2,
    markersize=7
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()








data["Cluster"]= kmeans.fit_predict(X)

print("\nCluster labels:")
print(data[["AttendancePercent" ,"CGPA","Cluster"]].head(9))

print("\nCluster Centers")
print(kmeans.cluster_centers_)

print("\n Number of student in each cluster ")
print(data["Cluster"].value_counts().sort_index())

plt.scatter(
    X["AttendancePercent"],
    X["CGPA"],
    c=data["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.75,
    edgecolors="black",
    linewidth=0.3
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids",
    edgecolors="black",
    linewidth=1.2
)

plt.xlabel("Attendance Percent")
plt.ylabel("CGPA")
plt.title("K-means Clustering")
plt.colorbar(label="Cluster")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()