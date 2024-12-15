from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
texts = [
    "A great game",
    "The election was over",
    "Very clean match",
    "A clean but forgettable game",
    "It was a close election",
    "A good writer",


]
tags = ["Sports", "Not sports", "Sports", "Sports", "Not sports","Not sports"]

# New sentence to classify
new_sentence = "A very good reader"

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)
y_train = tags

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Vectorize the new sentence
X_test = vectorizer.transform([new_sentence])

# Predict the tag for the new sentence
prediction = classifier.predict(X_test)

print(f"The sentence '{new_sentence}' is classified as '{prediction[0]}'.")
