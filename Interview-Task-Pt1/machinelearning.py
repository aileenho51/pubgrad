# Experimenting with Machine Learning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Load and preprocess data
def load_data():
    df = pd.read_csv("masterdata.csv").dropna()
    return df

# Apply linear regression and visualize results
# Parameters: DataFrame
# Return: None
def plot_linear_regression(df):
    X = df[["1st Order Profit"]]
    y = df["Subsequent Orders Count"]
    model = LinearRegression().fit(X, y)

    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color="red")
    plt.xlabel("1st Order Profit")
    plt.ylabel("Subsequent Orders Count")
    plt.title("Linear Regression")
    plt.savefig("/Users/aileen.ho/PycharmProjects/PythonProject/ml_linear_regression.png", dpi=300)
    plt.show()

# Apply K-Means clustering and visualize segments with percentage breakdown
# Parameters: DataFrame
# Return: None
def plot_kmeans_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(
        df[["1st Order Profit", "Subsequent Orders Count"]]
    )
    df["Customer Segment"] = kmeans.labels_

    segment_names = {
        0: "One-Time Shoppers",      # Low profit, low repeat purchases
        1: "Loyal Repeat Buyers",    # Moderate profit, high repeat purchases
        2: "High-Value Customers"    # High profit, varied repeat behavior
    }
    df["Segment Label"] = df["Customer Segment"].map(segment_names)

    segment_counts = df["Segment Label"].value_counts(normalize=True) * 100

    print("Customer Segment Distribution (%):")
    print(segment_counts)

    plt.figure(figsize=(8, 6))

    # Plot each segment separately
    for segment in segment_names.keys():
        subset = df[df["Customer Segment"] == segment]
        plt.scatter(
            subset["1st Order Profit"],
            subset["Subsequent Orders Count"],
            label=segment_names[segment],
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