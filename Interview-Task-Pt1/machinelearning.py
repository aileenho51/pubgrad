# Experimenting with Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# Load and preprocess data
def load_data():
    df = pd.read_csv("masterdata.csv").dropna()
    return df

# Apply linear regression and visualize results
# Parameters: DataFrame
# Return: None
def plot_linear_regression(df):
    X = df[["1st Order Profit"]].values
    y = df["Subsequent Orders Count"].values

    model = LinearRegression().fit(X, y)

    # sort values for a cleaner regression line
    order = np.argsort(X[:, 0])
    X_sorted, y_sorted = X[order], y[order]

    plt.scatter(X_sorted, y_sorted, alpha=0.5)
    plt.plot(X_sorted, model.predict(X_sorted), color="red")
    plt.xlabel("1st Order Profit")
    plt.ylabel("Subsequent Orders Count")
    plt.title("Linear Regression")
    plt.savefig("ml_linear_regression.png", dpi=300)
    plt.show()

# Apply K-Means clustering and visualize segments with percentage breakdown
# Parameters: DataFrame
# Return: None
def plot_kmeans_clustering(df):
    features = ["1st Order Profit", "Subsequent Orders Count"]

    # scale features for balanced clustering
    pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42, n_init=50))
    pipe.fit(df[features])

    # get cluster centers back in original units
    scaler = pipe.named_steps["standardscaler"]
    kmeans = pipe.named_steps["kmeans"]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=features)
    centers_df["Cluster"] = range(3)

    # map clusters to human-readable segment names based on center values
    by_orders = centers_df.sort_values("Subsequent Orders Count")
    one_time = by_orders.iloc[0]["Cluster"]
    remaining = by_orders.iloc[1:][["Cluster", "1st Order Profit"]].sort_values("1st Order Profit")
    loyal = remaining.iloc[0]["Cluster"]
    high_value = remaining.iloc[1]["Cluster"]

    segment_names = {
        int(one_time): "One-Time Shoppers",
        int(loyal): "Loyal Repeat Buyers",
        int(high_value): "High-Value Customers"
    }

    # assign labels to dataframe
    df["Customer Segment"] = pipe.predict(df[features])
    df["Segment Label"] = df["Customer Segment"].map(segment_names)

    # print segment distribution
    segment_counts = df["Segment Label"].value_counts(normalize=True) * 100
    print("Customer Segment Distribution (%):")
    print(segment_counts.round(1))

    # plot clusters
    plt.figure(figsize=(8, 6))
    for segment in segment_names.values():
        subset = df[df["Segment Label"] == segment]
        plt.scatter(
            subset["1st Order Profit"],
            subset["Subsequent Orders Count"],
            label=segment,
            alpha=0.6
        )

    plt.xlabel("1st Order Profit")
    plt.ylabel("Subsequent Orders Count")
    plt.title("K-Means Clustering")
    plt.legend(title="Customer Segment", loc="upper right")
    plt.tight_layout()
    plt.savefig("ml_kmeans_clustering.png", dpi=300)
    plt.show()

# Main function
def main():
    df = load_data()
    plot_linear_regression(df)
    plot_kmeans_clustering(df)

if __name__ == "__main__":
    main()
