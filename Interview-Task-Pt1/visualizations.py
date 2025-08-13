# Combined Analysis: Signups, Customer Demographics, and Trends

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read and process customer data
# Parameters: None
# Returns: Pandas DataFrame with cleaned and formatted customer data
def read_customer_data():
    file_path = "masterdata.csv"
    df = pd.read_csv(file_path, parse_dates=["Registration Date"])

    # Ensure 'Age' column exists
    if "Age" not in df.columns:
        raise ValueError("The dataset does not contain an 'Age' column.")

    # Convert Registration Date to datetime and extract day of the week
    df["Registration Date"] = pd.to_datetime(df["Registration Date"], format="%m/%d/%Y", errors='coerce')
    df["Day of Week"] = df["Registration Date"].dt.day_name()

    return df

# Function to plot and save signup trends with a 7-day rolling average
# Parameters: df, directory to save the plot
# Returns: None
def plot_signup_trends(df, save_path):
    signup_counts = df["Registration Date"].value_counts().sort_index()
    signup_counts_smoothed = signup_counts.rolling(window=7, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(signup_counts_smoothed.index, signup_counts_smoothed.values, linestyle='-', linewidth=2, color='darkred')
    ax.set(title="Daily Signups Over Time (7-Day Rolling Average)", xlabel="Date", ylabel="Number of Signups")

    # Annotate the highest peak
    max_date = signup_counts_smoothed.idxmax()
    max_value = signup_counts_smoothed.max()
    ax.annotate(f'Peak: {int(max_value)} signups', xy=(max_date, max_value), xytext=(max_date, max_value + 50),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, color='black')

    ax.grid(True)
    plt.xticks(rotation=45)

    fig.savefig(os.path.join(save_path, "v_signup_trends.png"), dpi=300)
    plt.close()

# Function to plot and save customer age distribution
# Parameters: df, directory to save the plot
# Returns: None
def plot_age_distribution(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="blue", ax=ax)
    ax.set(title="Customer Age Distribution", xlabel="Age", ylabel="Number of Customers")

    fig.savefig(os.path.join(save_path, "v_age_distribution.png"), dpi=300)
    plt.close()

# Function to plot and save signups by day of the week
# Parameters: df, directory to save the plot
# Returns: None
def plot_signups_by_day(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="Day of Week",
                  order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                  palette="viridis", ax=ax)
    ax.set(title="Signups by Day of the Week", xlabel="Day of the Week", ylabel="Number of Signups")

    fig.savefig(os.path.join(save_path, "v_signups_by_day.png"), dpi=300)
    plt.close()

# Main function to execute the script
# Parameters: None
# Returns: None
def main():
    save_path = "/Users/aileen.ho/PycharmProjects/PythonProject"
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    # Read and process data
    df = read_customer_data()

    # Generate and save all plots
    plot_signup_trends(df, save_path)
    plot_age_distribution(df, save_path)
    plot_signups_by_day(df, save_path)

    print(f"Plots saved successfully in: {save_path}")

if __name__ == '__main__':
    main()