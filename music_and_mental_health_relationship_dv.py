import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
def load_data():
    df = pd.read_csv("mxmh_survey_results.csv")
    return df

# Data Preprocessing/Cleaning
def clean_data(df):
    # Drop rows with missing values in essential columns
    essential_cols = ['Age', 'Primary streaming service', 'While working',
                      'Instrumentalist', 'Composer', 'Foreign languages', 'Music effects']
    df = df.dropna(subset=essential_cols)

    # Fill missing values for BPM column with median
    df['BPM'] = df['BPM'].fillna(df['BPM'].median())

    # Remove unrealistic values in Age and Hours per day
    df = df[(df['Age'] >= 10) & (df['Age'] <= 100)]
    df = df[(df['Hours per day'] >= 0) & (df['Hours per day'] <= 24)]

    # Standardize categorical data to lowercase and strip whitespace
    categorical_cols = ['Primary streaming service', 'Fav genre', 'Music effects', 'Foreign languages']
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.lower().str.strip())

    # Map frequency columns to numerical values
    freq_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    genre_cols = [col for col in df.columns if col.startswith('Frequency')]
    df[genre_cols] = df[genre_cols].replace(freq_mapping)

    # Corrected bins and labels for Age Group categories
    bins = [0, 18, 25, 35, 50, 65, 100]  # Six edges for five categories
    labels = ['Teens', 'Young Adults', 'Adults', 'Mid-Age', 'Seniors', 'Elderly']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

    return df

# Helper function to save or replace plot
def save_or_replace_plot(filename):
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)
    plt.close()

# Age Distribution
def plot_age_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title("Age Distribution of Respondents")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()
    save_or_replace_plot("age_distribution.png")

# Primary Streaming Service Usage
def plot_streaming_service(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Primary streaming service', order=df['Primary streaming service'].value_counts().index)
    plt.title("Primary Streaming Service Used by Respondents")
    plt.xlabel("Streaming Service")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    save_or_replace_plot("streaming_service_usage.png")

# Daily Hours of Music
def plot_hours_per_day(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Hours per day'], bins=10, kde=True)
    plt.title("Distribution of Hours Spent Listening to Music Daily")
    plt.xlabel("Hours per Day")
    plt.ylabel("Frequency")
    plt.show()
    save_or_replace_plot("hours_per_day.png")

# Favorite Genre
def plot_favorite_genre(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='Fav genre', order=df['Fav genre'].value_counts().index)
    plt.title("Favorite Music Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.show()
    save_or_replace_plot("favorite_genre.png")

# Correlation Heatmap of Mental Health Factors
def plot_correlation_heatmap(df):
    mental_health_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    corr = df[mental_health_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Between Mental Health Factors")
    plt.show()
    save_or_replace_plot("correlation_heatmap.png")

# Distribution of Mental Health Scores
def plot_mental_health_distribution(df):
    mental_health_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(mental_health_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col} Scores")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    save_or_replace_plot("mental_health_distribution.png")

# Genre vs. BPM
def plot_genre_bpm(df):
    sns.catplot(
    data=df.sort_values("Fav genre"),
    x="Fav genre", y="BPM", kind="boxen",height=6, aspect=2,width = 0.5,showfliers=False, palette='mako')
    plt.xticks(rotation = 90)
    plt.title('Genre vs BPM')
    plt.ylim(50, 210)
    plt.show()
    save_or_replace_plot("genre_bpm.png")

# Genre vs Hours per Day
def plot_genre_hours_per_day(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Fav genre', y='Hours per day', data=df)
    plt.title("Favorite Genre vs Hours Spent Listening per Day")
    plt.xlabel("Favorite Genre")
    plt.ylabel("Hours per Day")
    plt.xticks(rotation=45)
    plt.show()
    save_or_replace_plot("genre_hours_per_day.png")

# Impact of Music on Mental Health Conditions
def plot_music_effects(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Music effects', data=df, palette='mako')
    plt.title("Effect of Music on Mental Health")
    plt.xlabel("Music Effects")
    plt.ylabel("Count")
    plt.show()
    save_or_replace_plot("music_effects.png")

# Streaming service preference by age group
def streaming_service_by_age_group(df):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='Age Group', hue='Primary streaming service', palette='mako')
    plt.title("Primary Streaming Service Preference by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.legend(title='Streaming Service', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    save_or_replace_plot("streaming_service_by_age_group.png")

# Analyze the impact of music on mental health across age groups
def music_effects_on_mental_health(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Age Group', hue='Music effects', palette='coolwarm')
    plt.title("Effect of Music on Mental Health by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.legend(title="Music Effects", loc='upper right')
    plt.show()
    save_or_replace_plot("music_effects_by_age_group.png")

# Music Genre Preferences by Age Group
def genre_preference_by_age_group(df):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, y='Fav genre', hue='Age Group', palette='coolwarm', order=df['Fav genre'].value_counts().index)
    plt.title("Favorite Music Genre by Age Group")
    plt.xlabel("Count")
    plt.ylabel("Favorite Genre")
    plt.show()
    save_or_replace_plot("genre_preference_by_age_group.png")

# Average Mental Health Scores by Age Group
def average_mental_health_by_age_group(df):
    mental_health_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    age_group_mental_health = df.groupby('Age Group')[mental_health_cols].mean()
    age_group_mental_health.plot(kind='bar', figsize=(12, 8))
    plt.title("Average Mental Health Scores by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Average Score")
    plt.show()
    save_or_replace_plot("average_mental_health_by_age_group.png")

# Daily Music Listening Hours by Age Group
def listening_hours_by_age_group(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Age Group', y='Hours per day', palette='coolwarm')
    plt.title("Daily Music Listening Hours by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Hours per Day")
    plt.show()
    save_or_replace_plot("listening_hours_by_age_group.png")

# Most Popular Genres
def most_popular_genres(df):
    plt.figure(figsize=(12, 6))
    genre_counts = df['Fav genre'].value_counts().head(10)  # Top 10 genres
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='mako')
    plt.title("Top 10 Most Popular Music Genres")
    plt.xlabel("Number of Respondents")
    plt.ylabel("Genre")
    plt.show()
    save_or_replace_plot("most_popular_genres.png")

    print("Top 10 Most Popular Genres:\n", genre_counts)

# Genre-Based Mental Health Insights
def genre_mental_health_insights(df):
    popular_genres = df['Fav genre'].value_counts().index[:5]  # Top 5 genres
    mental_health_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

    plt.figure(figsize=(12, 8))
    for i, genre in enumerate(popular_genres, 1):
        plt.subplot(2, 3, i)
        genre_df = df[df['Fav genre'] == genre]
        sns.boxplot(data=genre_df[mental_health_cols])
        plt.title(f"Mental Health Scores for {genre}")
        plt.ylim(0, 10)
    plt.suptitle("Mental Health Scores by Popular Genre", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    save_or_replace_plot("genre_mental_health_insights.png")

# Genre Listening Frequency
def genre_listening_frequency(df):
    frequency_cols = [col for col in df.columns if col.startswith('Frequency')]
    popular_genre_freq = df[frequency_cols].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=popular_genre_freq.values, y=popular_genre_freq.index, palette='mako')
    plt.title("Listening Frequency of Popular Genres")
    plt.xlabel("Average Listening Frequency")
    plt.ylabel("Genre")
    plt.show()
    save_or_replace_plot("genre_listening_frequency.png")

    print("Listening Frequency for Popular Genres:\n", popular_genre_freq)

# Genre Popularity by Age Group
def genre_popularity_by_age_group(df):
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df, y='Fav genre', hue='Age Group',
                  order=df['Fav genre'].value_counts().head(10).index, palette='coolwarm')
    plt.title("Top 10 Favorite Genres by Age Group")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.legend(title="Age Group")
    plt.show()
    save_or_replace_plot("genre_popularity_by_age_group.png")


# Plots line charts for mental health factors (Insomnia, OCD, Depression, Anxiety) across favorite genres
def plot_genre_vs_mental_health(df):

    fig, axes = plt.subplots(2, 2, figsize=(30, 15))

    # Plot Insomnia by Favorite Genre
    sns.lineplot(ax=axes[0, 0], x=df['Fav genre'], y=df['Insomnia'], ci=None, color='#4D194D')
    axes[0, 0].set_title("Insomnia vs Favorite Genre")

    # Plot OCD by Favorite Genre
    sns.lineplot(ax=axes[0, 1], x=df['Fav genre'], y=df['OCD'], ci=None, color='#272640')
    axes[0, 1].set_title("OCD vs Favorite Genre")

    # Plot Depression by Favorite Genre
    sns.lineplot(ax=axes[1, 0], x=df['Fav genre'], y=df['Depression'], ci=None, color='#1B3A4B')
    axes[1, 0].set_title("Depression vs Favorite Genre")

    # Plot Anxiety by Favorite Genre
    sns.lineplot(ax=axes[1, 1], x=df['Fav genre'], y=df['Anxiety'], ci=None, color='#006466')
    axes[1, 1].set_title("Anxiety vs Favorite Genre")

    # Adjust layout and rotate x-axis labels
    plt.tight_layout()
    for ax in axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    # Save and replace the plot
    plt.show()
    save_or_replace_plot("genre_vs_mental_health.png")

# Main function with menu for user
def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)

    while True:
        print("\nChoose an option for analysis:")
        print("1. Age Distribution")
        print("2. Primary Streaming Service Usage")
        print("3. Hours of Music per Day")
        print("4. Favorite Music Genre")
        print("5. Correlation Heatmap of Mental Health Factors")
        print("6. Distribution of Mental Health Scores")
        print("7. Genre vs BPM")
        print("8. Genre vs Hours per Day")
        print("9. Impact of Music on Mental Health")
        print("10. Genre vs mental health")
        print("11. Streaming Service Preference by Age Group")
        print("12. Music Effects on Mental Health Across Age Groups")
        print("13. Music Genre Preference by Age Group")
        print("14. Average Mental Health Scores by Age Group")
        print("15. Daily Music Listening Hours by Age Group")
        print("16. Most Popular Music Genres")
        print("17. Mental Health Insights by Popular Genre")
        print("18. Genre Listening Frequency")
        print("19. Genre Popularity by Age Group")
        print("20. Exit")

        # Take user input
        choice = input("Enter your choice (1-20): ")

        # Use if-else to call the correct function based on user input
        if choice == '1':
            plot_age_distribution(df)
        elif choice == '2':
            plot_streaming_service(df)
        elif choice == '3':
            plot_hours_per_day(df)
        elif choice == '4':
            plot_favorite_genre(df)
        elif choice == '5':
            plot_correlation_heatmap(df)
        elif choice == '6':
            plot_mental_health_distribution(df)
        elif choice == '7':
            plot_genre_bpm(df)
        elif choice == '8':
            plot_genre_hours_per_day(df)
        elif choice == '9':
            plot_music_effects(df)
        elif choice == '10':
            plot_genre_vs_mental_health(df)
        elif choice == '11':
            streaming_service_by_age_group(df)
        elif choice == '12':
            music_effects_on_mental_health(df)
        elif choice == '13':
            genre_preference_by_age_group(df)
        elif choice == '14':
            average_mental_health_by_age_group(df)
        elif choice == '15':
            listening_hours_by_age_group(df)
        elif choice == '16':
            most_popular_genres(df)
        elif choice == '17':
            genre_mental_health_insights(df)
        elif choice == '18':
            genre_listening_frequency(df)
        elif choice == '19':
            genre_popularity_by_age_group(df) 
        elif choice == '20':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 20.")

# Run the main function
if __name__ == "__main__":
    main()

