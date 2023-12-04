import customtkinter as ctk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from PIL import Image

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
appWidth, appHeight = 300, 500


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("K-means Clustering")
        self.geometry(f"{appWidth}x{appWidth}")

        current_path = os.path.dirname(os.path.realpath(__file__))
        self.bg_image = ctk.CTkImage(Image.open(current_path + "/bg.jpg"),
                                               size=(300, 400))

        self.bg_image_label = ctk.CTkLabel(self, image=self.bg_image, text=None)

        self.bg_image_label.place(x=0, y=0)

        self.elbowcurve = ctk.CTkButton(self,
                                        text="Elbow Curve",
                                        command=self.elbowcurve)

        self.elbowcurve.grid(row=5, column=1,
                                        columnspan=2,
                                        padx=20, pady=20,
                                        sticky="ew")

        self.twodimresutls = ctk.CTkButton(self,
                                        text="2D Classification",
                                        command=self.twodimresutls )

        self.twodimresutls .grid(row=6, column=1,
                             columnspan=2,
                             padx=20, pady=20,
                             sticky="ew")

        self.threedimresutls = ctk.CTkButton(self,
                                           text="3D Classification",
                                           command=self.threedimresutls)

        self.threedimresutls.grid(row=7, column=1,
                                columnspan=2,
                                padx=20, pady=20,
                                sticky="ew")

        self.result = ctk.CTkButton(self,
                                             text="Generate List",
                                             command=self.result)

        self.result.grid(row=8, column=1,
                                  columnspan=2,
                                  padx=20, pady=20,
                                  sticky="ew")

    def result(self):
        dataset = pd.read_csv("Mall_Customers.csv")

        X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]
        kmeans = KMeans(n_clusters=5)
        y = kmeans.fit_predict(X)
        dataset['Cluster'] = y

        File = pd.concat([dataset['CustomerID'], dataset['Cluster']], axis=1)
        f = pd.DataFrame(File)
        f.to_csv('Result.csv')

    def elbowcurve(self):
        dataset = pd.read_csv("Mall_Customers.csv")

        X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

        wcss = []
        for i in range(1, 11):
            k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
            k_means.fit(X)
            wcss.append(k_means.inertia_)

        plt.plot(np.arange(1, 11), wcss)
        plt.xlabel('Clusters')
        plt.ylabel('SSE')
        plt.show()

    def twodimresutls(self):
        dataset = pd.read_csv("Mall_Customers.csv")

        X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

        kmeans = KMeans(n_clusters=5)

        y = kmeans.fit_predict(X)

        dataset['Cluster'] = y

        pca_num_components = 2

        reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
        results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

        sns.scatterplot(x="pca1", y="pca2", hue=dataset['Cluster'], data=results)
        plt.title('K-means Clustering')
        plt.show()

    def threedimresutls(self):
        dataset = pd.read_csv("Mall_Customers.csv")

        X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

        kmeans = KMeans(n_clusters=5)

        y = kmeans.fit_predict(X)

        dataset['Cluster'] = y

        kplot = plt.axes(projection='3d')
        xline = np.linspace(0, 15, 1000)
        yline = np.linspace(0, 15, 1000)
        zline = np.linspace(0, 15, 1000)

        data1 = dataset[dataset.Cluster == 0]
        data2 = dataset[dataset.Cluster == 1]
        data3 = dataset[dataset.Cluster == 2]
        data4 = dataset[dataset.Cluster == 3]
        data5 = dataset[dataset.Cluster == 4]

        kplot.scatter3D(data1['Annual Income (k$)'], data1['Spending Score (1-100)'], data1['Age'], c='red',
                        label='Cluster 1')
        kplot.scatter3D(data2['Annual Income (k$)'], data2['Spending Score (1-100)'], data2['Age'], c='green',
                        label='Cluster 2')
        kplot.scatter3D(data3['Annual Income (k$)'], data3['Spending Score (1-100)'], data3['Age'], c='blue',
                        label='Cluster 3')
        kplot.scatter3D(data4['Annual Income (k$)'], data4['Spending Score (1-100)'], data4['Age'], c='yellow',
                        label='Cluster 4')
        kplot.scatter3D(data5['Annual Income (k$)'], data5['Spending Score (1-100)'], data5['Age'], c='pink',
                        label='Cluster 5')

        kplot.set_xlabel('Annual Income (k$)', labelpad=20)
        kplot.set_ylabel('Spending Score (1-100)', labelpad=20)
        kplot.set_zlabel('Age', labelpad=20)

        plt.legend(loc='upper right')
        plt.title("K-means Clustering")
        plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()