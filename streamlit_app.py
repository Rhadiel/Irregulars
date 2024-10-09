import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration for better layout and appearance
st.set_page_config(page_title="Wine Quality Analysis", layout="wide", page_icon="🍷")

# --------------------------
# Introduction Section
# --------------------------
def introduction():
    st.title("🍷 Wine Quality Data Analysis")
    st.header("Introduction")
    st.markdown("""
    The **Wine Quality** dataset contains various chemical and sensory properties of red and white wine samples, along with their quality ratings. This dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

    ### Key Columns Affecting Wine Quality:
    1. **Fixed Acidity**: Measures the non-volatile acids in wine, such as tartaric, malic, and citric acids. High levels contribute to a crisp or fresh taste.
    2. **Volatile Acidity**: Represents the amount of acetic acid in wine, which can give vinegar-like flavors. Excessive volatile acidity can spoil the wine’s taste, making it a critical quality indicator.
    3. **Citric Acid**: Adds freshness to the wine. Higher concentrations are generally found in higher-quality wines.
    4. **Alcohol**: Influences the perception of the wine’s body and flavor. Higher alcohol content often correlates with better wine quality.
    5. **Sulphates**: Acts as a preservative and helps stabilize the wine. However, too much sulphate can cause bitterness.

    ### Purpose of Exploration:
    The primary goal of this exploration is to identify and understand the key factors that significantly affect the quality of wine. By analyzing these factors, we aim to derive insights that can help in enhancing wine production and quality control processes.
    """)

# --------------------------
# Data Cleaning and Describing Section
# --------------------------
def data_cleaning_and_describing(df, df_cleaned):
    st.header("📊 Data Cleaning and Descriptive Statistics")

    # Create tabs for Raw Data, Data Cleaning, and Descriptive Statistics
    tab1, tab2, tab3 = st.tabs(["📂 Raw Data", "🧹 Data Cleaning", "📈 Descriptive Statistics"])

    with tab1:
        st.subheader("📂 Raw Data")
        st.write("Here is a preview of the raw dataset:")
        st.dataframe(df.head())  # Display the first few rows of the raw data

    with tab2:
        st.subheader("🧹 Data Cleaning Process")
        st.markdown("""
        The data cleaning process involves several steps to ensure data integrity and reliability:
        - **Removing Missing Values:** Dropping any rows with missing data to prevent skewed analysis.
        - **Removing Duplicates:** Eliminating duplicate entries to ensure each data point is unique.
        - **Type Conversion:** Converting the 'Id' column to integer type for consistency.
        - **Filtering Outliers:** Keeping 'fixed acidity' values between 0 and 15 to remove unrealistic entries.
        - **Dropping Unnecessary Columns:** Removing the 'Id' column as it does not contribute to the analysis.
        """)
        st.write("### Missing Values in Each Column:")
        st.write(df.isnull().sum())  # Display the count of missing values per column

        st.write("### Cleaned Data Preview:")
        st.dataframe(df_cleaned.head())  # Display the first few rows of the cleaned data

    with tab3:
        st.subheader("📈 Descriptive Statistics")
        st.markdown("""
        Descriptive statistics provide a summary of the central tendency, dispersion, and shape of the dataset’s distribution. Below are the key statistical metrics for the cleaned dataset:
        """)
        
        # Display descriptive statistics
        desc_stats = df_cleaned.describe().T  # Transpose for better readability
        desc_stats['median'] = df_cleaned.median()
        desc_stats = desc_stats[['mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]
        st.dataframe(desc_stats.style.highlight_max(axis=0))  # Highlight maximum values for emphasis

        # Add explanations for each statistic
        st.markdown("""
        ### 📚 Understanding the Statistics:
        - **Mean:** The average value of each feature, providing a central value around which data points are distributed.
        - **Median:** The middle value, which helps in understanding the distribution and identifying skewness.
        - **Standard Deviation (std):** Measures the amount of variation or dispersion in the dataset.
        - **Minimum (min) & Maximum (max):** Indicate the range of the data, highlighting the spread.
        - **25% & 75% Percentiles:** Provide insights into the data distribution, showing where the bulk of the data points lie.
        """)

# --------------------------
# Data Visualization Section
# --------------------------
def data_visualization(df_cleaned):
    st.header("📊 Data Visualization")

    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Correlation Heatmap", "📦 Boxplot", "📈 Quality Distribution", "🔍 Pairplot"])

    with tab1:
        st.subheader("🔗 Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        corr = df_cleaned.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
        st.pyplot(plt.gcf())  # Display the correlation heatmap
        st.markdown("""
        The **Correlation Heatmap** visualizes the strength and direction of relationships between different features:
        - **Positive Correlation:** Features like 'alcohol' and 'quality' show a strong positive correlation, indicating that higher alcohol content tends to improve wine quality.
        - **Negative Correlation:** Features such as 'volatile acidity' and 'quality' exhibit a negative correlation, suggesting that higher volatile acidity may reduce wine quality.
        """)

        plt.clf()  # Clear the figure to prevent overlap

    with tab2:
        st.subheader("📦 Boxplot of Key Features")
        # Allow users to select features for the boxplot
        selected_features = st.multiselect(
            "Select features to display boxplots:",
            options=df_cleaned.columns.tolist(),
            default=['fixed acidity', 'volatile acidity', 'quality']
        )
        if selected_features:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=df_cleaned[selected_features], palette="Set3")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())  # Display the boxplot
            st.markdown("""
            **Boxplots** help identify the distribution and potential outliers in the selected features:
            - **Outliers:** Points outside the whiskers may indicate anomalies or variations worth investigating.
            - **Interquartile Range (IQR):** Shows the middle 50% of the data, giving insights into data spread and central tendency.
            """)

            plt.clf()  # Clear the figure to prevent overlap

    with tab3:
        st.subheader("📈 Distribution of Wine Quality")
        # Allow users to select which quality metric to visualize
        selected_quality = st.selectbox(
            "Select a quality feature to visualize:",
            options=['quality']  # Assuming 'quality' is the relevant feature
        )
        if selected_quality:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_cleaned[selected_quality], kde=True, bins=10, color='teal')
            plt.title(f"Distribution of {selected_quality.capitalize()}")
            st.pyplot(plt.gcf())  # Display the distribution plot
            st.markdown("""
            The **Distribution Plot** shows how wine quality ratings are spread across different scores:
            - **Peaks:** Indicate the most common quality ratings.
            - **Skewness:** If the distribution is skewed, it suggests that most wines fall on one side of the quality spectrum.
            """)

            plt.clf()  # Clear the figure to prevent overlap

    with tab4:
        st.subheader("🔍 Pairplot of Key Features")
        # Allow users to select features for the pairplot
        pairplot_features = st.multiselect(
            "Select features for pairplot:",
            options=df_cleaned.columns.tolist(),
            default=['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol', 'quality']
        )
        if pairplot_features:
            plt.figure(figsize=(12, 10))
            sns.pairplot(df_cleaned[pairplot_features], hue='quality', palette='viridis', diag_kind='kde')
            st.pyplot(plt.gcf())  # Display the pairplot
            st.markdown("""
            **Pairplots** help visualize relationships between multiple features simultaneously:
            - **Diagonal:** Shows the distribution of individual features.
            - **Off-Diagonal:** Illustrates pairwise relationships, highlighting how two features interact and relate to wine quality.
            """)

            plt.clf()  # Clear the figure to prevent overlap

# --------------------------
# Conclusion Section
# --------------------------
def conclusion():
    st.header("✅ Conclusion and Key Findings")
    st.markdown("""
    ### 🔑 Key Findings
    
    1. **Correlations between Variables:**
        - **Positive Correlation:** 'Alcohol' shows a strong positive correlation with 'quality', suggesting that higher alcohol content tends to improve wine quality.
        - **Negative Correlation:** 'Volatile acidity' is negatively correlated with 'quality', indicating that higher levels of volatile acidity may reduce wine quality.
    
    2. **Outliers Detected:**
        - **Volatile Acidity:** The boxplot reveals several outliers, which could represent wines with unusually high acidity levels, potentially affecting their quality.
        - **Fixed Acidity:** No extreme outliers were observed within the range (0 < fixed acidity < 15), ensuring data reliability for this feature.
    
    3. **Wine Quality Distribution:**
        - Most wines in the dataset have quality ratings between 5 and 7.
        - The distribution is slightly skewed towards lower quality ratings, with fewer wines rated 8 or above.
    
    4. **Relationships between Key Features:**
        - Wines with lower volatile acidity and higher fixed acidity generally tend to have higher quality ratings, as seen in the pairplot.
    
    ### 📝 Conclusion
    
    This analysis provided valuable insights into the factors influencing wine quality. By identifying key correlations and understanding the distribution of quality ratings, stakeholders can make informed decisions to enhance wine production and quality control processes. Future analyses could explore more advanced modeling techniques to predict wine quality based on the identified factors.
    """)

# --------------------------
# Main Application
# --------------------------
def main():
    # Load the dataset
    @st.cache_data  # Cache the data loading for efficiency
    def load_data():
        return pd.read_csv('https://raw.githubusercontent.com/jessrey679/IndustryElective/refs/heads/main/WineQT.csv')
    
    df = load_data()

    # Sidebar Navigation with Icons for better UI
    st.sidebar.header("🗂️ Navigation")
    option = st.sidebar.radio(
        "Go to",
        ['Introduction', 'Data Cleaning & Describing', 'Data Visualization', 'Conclusion']
    )

    # Sidebar Filters (Optional Enhancements)
    st.sidebar.header("🔍 Filters")
    
    # Example: Filter by Quality
    quality_min, quality_max = int(df['quality'].min()), int(df['quality'].max())
    quality_filter = st.sidebar.slider(
        "Select Quality Range:",
        min_value=quality_min,
        max_value=quality_max,
        value=(quality_min, quality_max)
    )
    
    # Example: Filter by Alcohol Content
    alcohol_min, alcohol_max = float(df['alcohol'].min()), float(df['alcohol'].max())
    alcohol_filter = st.sidebar.slider(
        "Select Alcohol Content Range:",
        min_value=alcohol_min,
        max_value=alcohol_max,
        value=(alcohol_min, alcohol_max)
    )
    
    # Apply Filters
    df_filtered = df[
        (df['quality'] >= quality_filter[0]) & (df['quality'] <= quality_filter[1]) &
        (df['alcohol'] >= alcohol_filter[0]) & (df['alcohol'] <= alcohol_filter[1])
    ]

    # Sidebar Download Button
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Data")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_wine_quality.csv',
        mime='text/csv',
    )

    # Data Cleaning and Caching
    if 'df_cleaned' not in st.session_state:
        # Clean the data
        df_cleaned = df_filtered.dropna().drop_duplicates()  # Remove missing values and duplicates
        df_cleaned['Id'] = df_cleaned['Id'].astype(int)  # Convert 'Id' to integer
        # Filter 'fixed acidity' to be within realistic bounds
        df_cleaned = df_cleaned[(df_cleaned['fixed acidity'] > 0) & (df_cleaned['fixed acidity'] < 15)]
        df_cleaned = df_cleaned.drop(columns=['Id'])  # Drop the 'Id' column as it's not needed
        st.session_state.df_cleaned = df_cleaned  # Store cleaned data in session state
    
    df_cleaned = st.session_state.df_cleaned

    # Navigation Options
    if option == 'Introduction':
        introduction()
    elif option == 'Data Cleaning & Describing':
        data_cleaning_and_describing(df, df_cleaned)
    elif option == 'Data Visualization':
        data_visualization(df_cleaned)
    elif option == 'Conclusion':
        conclusion()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 About")
    st.sidebar.markdown("""
    Developed by Irregulars 
    [GitHub Repository](https://github.com/your-repo)  
    For more information, visit the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/rajyellow46/wine-quality).
    """)

# Run the app
if __name__ == "__main__":
    main()
