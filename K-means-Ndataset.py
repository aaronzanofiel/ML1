import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data():
    data=pd.read_csv("placement_predict_50k Dataset (3)(in).csv")
    x=data[["AttendancePercent","CGPA"]]
    return x


def manual_k():
    print("\nManual k selection ---- ")
    k=int(input("enter the number of k:"))
    while k<2:
        print("K must be atleast 2!")
        k=int(input("enter the number of k:"))
    return k

def elbow_method():
    x=load_data()
    k_values=range(2,11)
    wcss=[]


    for k in k_values:
        kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

        print("\n WCSS values:")

    for k,value in zip(k_values,wcss):
        print("K =",k,"WCSS =",value)

    plt.plot(k_values,wcss,marker='o')

    plt.xlabel("No.of clusters")
    plt.ylabel("WCSS value")
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

    k=int(input("enter the number of k based on elbow graph:"))
    return k

def silhouette_method():
    x=load_data()
    k_values=range(2,11)
    scores=[]

    sample_size=5000
    for k in k_values:
        print("Calculating for k=",k)
        kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)

        labels=kmeans.fit_predict(x)
        score = silhouette_score(x,labels,sample_size=sample_size,random_state=42)
        scores.append(score)
        print("Silhouette score =",score)

    #find highest score
    best_index=scores.index(max(scores))
    best_k = list(k_values)[best_index]

    print("\n Highest silhouette score=",max(scores))
    print("Recommended k=",best_k)

    plt.plot(k_values,scores,marker='o')

    plt.xlabel("No.of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score method")
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

    return best_k

def select_method():
    print("\nSelect method:")
    print("1.   Manual k selection ")
    print("2.   Elbow method")
    print("3.   Silhouette method")

    choice=int(input("enter your choice(1/2/3):"))
    return choice


def main():
    choice=select_method()
    x=load_data()

    #call only the module selected by the user
    if choice==1:
        k=manual_k()
    elif choice==2:
        k=elbow_method()
    elif choice==3:
        k=silhouette_method()
    else:
        print("INVALID choice")

    return k

if __name__=="__main__":
    k=main()