import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.summarization.keypoints import keywords
from gensim.summarization import summarize
from sklearn.decomposition import LatentDirichletAllocation

# Load the transcript data
transcript_data = "to-transcript"

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

preprocessed_data = [preprocess_text(text) for text in transcript_data]

# Perform topic modeling using LDA
lda_model = LatentDirichletAllocation(n_topics=5, max_iter=5)
topics = lda_model.fit_transform(preprocessed_data)

# Generate summary using TextRank
summary = summarize(preprocessed_data, ratio=0.2)

# Extract key points using NER
entities = []
for sentence in preprocessed_data:
    entities.extend(keywords(sentence).split(', '))

print("Summary:", summary)
print("Key Points:", entities)