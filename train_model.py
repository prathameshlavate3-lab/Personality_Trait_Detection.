import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# DEMO DATA ONLY. These example scores are invented for testing the interface.
# Replace them with real labelled data for valid model evaluation.
data = [
    ("I love exploring new ideas and creative experiences.", [90, 55, 60, 65, 35]),
    ("I carefully plan my work and finish tasks on time.", [55, 95, 45, 70, 30]),
    ("I enjoy meeting people and talking in groups.", [60, 60, 95, 75, 40]),
    ("I help others and cooperate with my teammates.", [65, 70, 65, 95, 30]),
    ("I often feel worried and nervous about problems.", [45, 55, 40, 55, 90]),
    ("I enjoy learning, planning, meeting friends, and helping people.", [85, 85, 80, 90, 45]),
    ("I prefer routines, quiet time, and working alone.", [35, 80, 20, 60, 45]),
    ("I stay calm, organize tasks, and enjoy learning new things.", [80, 90, 50, 75, 20]),
    ("I like parties, meeting new people, and trying new activities.", [85, 50, 95, 70, 45]),
    ("I become stressed easily but try to support my friends.", [55, 60, 45, 90, 85]),
    ("I am curious, responsible, sociable, kind, and usually calm.", [90, 90, 85, 90, 20]),
    ("I avoid new experiences and dislike working with others.", [20, 35, 20, 20, 65]),
    ("I work hard, enjoy teamwork, and try to remain relaxed.", [60, 90, 70, 85, 20]),
    ("I like art, reading, and discovering different perspectives.", [95, 60, 45, 70, 35]),
    ("I worry about deadlines and sometimes struggle to relax.", [45, 70, 40, 60, 85]),
]

texts = [item[0] for item in data]
scores = [item[1] for item in data]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("regressor", MultiOutputRegressor(Ridge()))
])

model.fit(texts, scores)
joblib.dump(model, "personality_model.joblib")

print("Demo model saved successfully.")
print("Output order: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism")
print("IMPORTANT: The example scores are invented and are not scientifically validated.")