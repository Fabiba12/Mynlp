# Import the necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Download all the required nltk stopwords
nltk.download('stopwords')

# This is our first function which loads data from our CSV file given the extent of the file we adjust to only work on 1500 rows the number of rows can be changed based on preferences
def load_data(file_path, num_rows=1500):
    # Next step is reading the first 1500 rows of the CSV file into a DataFrame
    df = pd.read_csv(file_path, nrows=num_rows)
    return df

# This is the function for cleaning the data
def clean_text(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return ""
    # It starts by removing all punctuations from the text
    text = re.sub(r'[^\w\s]', '', text)
    # It then proceeds to convert the text to lowercase
    text = text.lower()
    # It then removes all the stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# This is our function for thematic analysis and displaying the key topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# This is our main function
def main():
    # The path for calling our CSV file
    file_path = 'Consumer_Complaints.csv'
    
    # This function loads the data to a dataframe
    df = load_data(file_path, num_rows=1500)

    # Check the first ten rows to obtain insight on our data
    print("DataFrame Head:")
    print(df.head())
    
    # Ensure the DataFrame has a column for comments, in this case our column is the 'Consumer complaint narrative' column
    if 'Consumer complaint narrative' not in df.columns:
        raise ValueError("The CSV file must contain a column for the consumer complaints.")
    
    # We clean the data by calling our clean text function
    df['cleaned_comments'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # We display the first 10 clean data for review
    print("Cleaned Data:")
    print(df['cleaned_comments'].head(10))
    
    # We complete vectorization using Bag of Words approach
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(df['cleaned_comments'])
    
    # We conduct vectorization using TF-IDF approach
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(df['cleaned_comments'])
    
    # We conduct topic modeling using LDA coupled with the vectorized data from Bag of Words
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(X_bow)
    
    # We display LDA topics based on the Bag of Words approach
    print("\nLDA Topics (Bag of Words):")
    display_topics(lda, vectorizer_bow.get_feature_names_out(), 5)

    # We conduct LDA using the vectorized data from TF-IDF approach
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    lda.fit(X_tfidf)

    # We display LDA topics based on the TF-IDF approach
    print("\nLDA Topics (TF-IDF):")
    display_topics(lda, vectorizer_tfidf.get_feature_names_out(), 5)

    # We conduct topic modeling using NMF with TF-IDF
    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(X_tfidf)
    
    # Display NMF topics
    print("\nNMF Topics (TF-IDF):")
    display_topics(nmf, vectorizer_tfidf.get_feature_names_out(), 5)

    # We conduct topic modeling using NMF with Bag of Words
    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(X_bow)
    
    # Display NMF topics
    print("\nNMF Topics (Bag of Words):")
    display_topics(nmf, vectorizer_bow.get_feature_names_out(), 5)

# This is our main function to run the code
if __name__ == "__main__":
    main()
