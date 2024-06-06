import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
import os

nltk.download('punkt')
nltk.download('stopwords')


def load_master_dictionary(positive_file, negative_file):
    
    positive_words = set()
    negative_words = set()
 
    with open(positive_file, "r") as file:
        for line in file:
            positive_words.add(line.strip())
 
    with open(negative_file, "r") as file:
        for line in file:
            negative_words.add(line.strip())
    return positive_words, negative_words


def clean_text(text, stop_words_folder):

    stop_words = set()
    for file_name in os.listdir(stop_words_folder):
        with open(os.path.join(stop_words_folder, file_name), "r") as file:
            for line in file:
                stop_words.add(line.strip())
   
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(clean_text.lower())
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Function to compute sentiment analysis
def compute_sentiment(text, positive_words, negative_words):

    positive_score = 0
    negative_score = 0
    polarity_score = 0

    for token in text:
        if token in positive_words:
            positive_score += 1
        elif token in negative_words:
            negative_score += 1
    
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score) + 0.000001
    return positive_score, negative_score, polarity_score


# Function to compute readability analysis
def compute_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    total_words = len(words)
    total_sentences = len(sentences)
    avg_sentence_length = total_words / total_sentences
    
    # Compute percentage of complex words
    complex_words = [word for word in words if textstat.syllable_count(word) > 2]
    percentage_complex_words = (len(complex_words) / total_words) * 100
    
    # Compute Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    # Compute average number of words per sentence
    avg_words_per_sentence = total_words / total_sentences
    
    # Compute syllable per word
    syllables_per_word = sum(textstat.syllable_count(word) for word in words) / total_words
    
    # Compute personal pronouns
    personal_pronouns = sum(1 for word in words if word.lower() in ['i', 'we', 'my', 'ours', 'us'])
    
    # Compute average word length
    avg_word_length = sum(len(word) for word in words) / total_words
    
    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length

df = pd.read_excel("Input.xlsx")
# Extract URLs from the DataFrame
urls = df["URL"].tolist()

positive_words, negative_words = load_master_dictionary("positive-words.txt", "negative-words.txt")
stop_words_folder = r"C:\Users\dibya\Blackcoffer_task\StopWords"

# Loop through each URL and update DataFrame with computed values
for i, url in enumerate(urls):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract title
        title_element = soup.find("h1", class_="entry-title")
        if title_element:
            title = title_element.get_text().strip()
        else:
            title = "Title not found"
        # Extract article content
        article_content = soup.find("article")
        if article_content:
    
            article_text = article_content.get_text()
            # Clean and tokenize text
            tokens = clean_text(article_text, stop_words_folder)
            
            # Sentiment analysis
            positive_score, negative_score, polarity_score = compute_sentiment(tokens, positive_words, negative_words)
            
            # Readability analysis
            avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length = compute_readability(article_text)
            
            # Update DataFrame with computed values
            df.at[i, "POSITIVE SCORE"] = positive_score
            df.at[i, "NEGATIVE SCORE"] = negative_score
            df.at[i, "POLARITY SCORE"] = polarity_score
            df.at[i, "AVG SENTENCE LENGTH"] = avg_sentence_length
            df.at[i, "PERCENTAGE OF COMPLEX WORDS"] = percentage_complex_words
            df.at[i, "FOG INDEX"] = fog_index
            df.at[i, "AVG NUMBER OF WORDS PER SENTENCE"] = avg_words_per_sentence
            df.at[i, "PERSONAL PRONOUNS"] = personal_pronouns
            df.at[i, "AVG WORD LENGTH"] = avg_word_length
        else:
            print("Article content not found for URL:", url)
    else:
        print("Error:", response.status_code)

# Save the DataFrame with computed values to a new CSV file
df.to_csv("output_with_computed_values.csv", index=False)
