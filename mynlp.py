# Import the necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# Download all the required nltk stopwords
nltk.download('stopwords')

# This is our first function which loads data from our CSV file
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

# Function to calculate coherence score for a given model
def calculate_coherence_score(model, texts, vectorizer, coherence='c_v'):
    topics = model.transform(vectorizer.transform(texts))
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=None, coherence=coherence)
    return cm.get_coherence()

# This is our main function
def main():
    # The path for calling our CSV file
    file_path = 'Consumer_Complaints.csv'
    
    # This function loads the data to a dataframe
    df = load_data(file_path, num_rows=1500)

    # Ensure the DataFrame has a column for comments, in this case our column is the 'Consumer complaint narrative' column
    if 'Consumer complaint narrative' not in df.columns:
        raise ValueError("The CSV file must contain a column for the consumer complaints.")
    
    # We clean the data by calling our clean text function
    df['cleaned_comments'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # We complete vectorization using Bag of Words approach
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(df['cleaned_comments'])
    
    # We conduct vectorization using TF-IDF approach
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(df['cleaned_comments'])
    
    # Initialize variables for coherence evaluation
    coherence_values_bow = []
    coherence_values_tfidf = []
    k_range = range(2, 10)  # Choose a range of K values to evaluate

    # Iterate over different number of topics (K)
    for k in k_range:
        # Perform LDA with K topics using Bag of Words
        lda_bow = LatentDirichletAllocation(n_components=k, random_state=0)
        lda_bow.fit(X_bow)
        
        # Calculate coherence score for LDA with Bag of Words
        coherence_bow = calculate_coherence_score(lda_bow, df['cleaned_comments'], vectorizer_bow)
        coherence_values_bow.append(coherence_bow)
        
        # Perform LDA with K topics using TF-IDF
        lda_tfidf = LatentDirichletAllocation(n_components=k, random_state=0)
        lda_tfidf.fit(X_tfidf)
        
        # Calculate coherence score for LDA with TF-IDF
        coherence_tfidf = calculate_coherence_score(lda_tfidf, df['cleaned_comments'], vectorizer_tfidf)
        coherence_values_tfidf.append(coherence_tfidf)

    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, coherence_values_bow, marker='o', label='Bag of Words')
    plt.plot(k_range, coherence_values_tfidf, marker='o', label='TF-IDF')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Score vs. Number of Topics')
    plt.legend()
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

# This is our main function to run the code
if __name__ == "__main__":
    main()
