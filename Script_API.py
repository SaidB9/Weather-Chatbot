
from fastapi import FastAPI
import joblib
import re

# 1. Initialisation de l'application
app = FastAPI(title="API Chatbot Météo - Projet UA3")

# 2. Chargement des fichiers créés à l'étape 3
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 3. Route d'accueil (pour vérifier que l'API fonctionne)
@app.get("/")
def home():
    return {"status": "En ligne", "message": "API de prédiction d'intentions météo prête."}

# 4. Route de prédiction (Livrable Step 4)
@app.get("/predict")
def predict(text: str):
    """
    Reçoit une phrase utilisateur et renvoie l'intention détectée.
    Exemple : /predict?text=What is the weather in Paris?
    """
    # Nettoyage identique à l'Étape 2 : minuscules et retrait ponctuation
    text_clean = text.lower()
    text_clean = re.sub(r'[^\w\s]', '', text_clean)
    
    # Transformation de la phrase en nombres (Vectorisation)
    text_tfidf = vectorizer.transform([text_clean])
    
    # Prédiction de l'intention (current_weather ou forecast_weather)
    prediction = model.predict(text_tfidf)
    
    # Réponse au format JSON demandée par le projet
    return {
        "user_phrase": text,
        "cleaned_phrase": text_clean,
        "predicted_intent": prediction
    }
