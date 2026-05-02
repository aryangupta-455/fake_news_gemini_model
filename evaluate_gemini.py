import os
import time
import pandas as pd
import google.generativeai as genai
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from tqdm import tqdm

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

# -----------------------------
# 2. Load Fake & True Dataset
# -----------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df[["text", "label"]]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 4. Automatically Limit Samples
# -----------------------------
MAX_SAMPLES = 200   # Safe limit

if len(test_df) > MAX_SAMPLES:
    test_df = test_df.sample(MAX_SAMPLES, random_state=42)

texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

# -----------------------------
# 5. Gemini Prediction Function
# -----------------------------
def get_prediction(text):
    prompt = f"""
    Classify the following news strictly as:
    Real
    Fake

    Return only one word: Real or Fake.

    News: {text}
    """

    try:
        response = model.generate_content(prompt)
        output = response.text.strip().lower()

        if "real" in output:
            return 1
        elif "fake" in output:
            return 0
        else:
            return None

    except Exception:
        return None


# -----------------------------
# 6. Run Evaluation
# -----------------------------
predictions = []

print(f"\nEvaluating on {len(texts)} samples...\n")

for text in tqdm(texts):
    pred = get_prediction(text)
    predictions.append(pred)
    time.sleep(1)  # Safe delay to avoid rate limit

# -----------------------------
# 7. Remove Failed Predictions
# -----------------------------
valid_indices = [i for i, p in enumerate(predictions) if p is not None]

filtered_true = [true_labels[i] for i in valid_indices]
filtered_pred = [predictions[i] for i in valid_indices]

# -----------------------------
# 8. Compute Metrics
# -----------------------------
accuracy = accuracy_score(filtered_true, filtered_pred)
report = classification_report(filtered_true, filtered_pred)

print("\n==============================")
print(" GEMINI ZERO-SHOT RESULTS ")
print("==============================\n")

print("Total evaluated samples:", len(filtered_true))
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(report)
