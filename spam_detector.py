import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["message"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# User input
msg = input("Enter SMS: ")
msg_vector = vectorizer.transform([msg])

# Prediction
result = model.predict(msg_vector)

if result[0] == 1:
    print("🚫 Spam Message")
else:
    print("✅ Not Spam Message")
