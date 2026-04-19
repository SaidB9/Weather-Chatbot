from fastapi import FastAPI
import joblib
import requests
import re

app = FastAPI()

# 1. Charger le modèle et le vectoriseur
model = joblib.load('model_meteo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_real_weather(city="Ottawa"):
    """Cherche la vraie température sur internet"""
    try:
        # On demande le format JSON et le français (en HTTP pour éviter les erreurs SSL Azure)
        url = f"http://wttr.in/{city}?format=j1&lang=fr"
        response = requests.get(url).json()
        
        # Extraction sécurisée des données
        temp = response['current_condition'][0]['temp_C']
        desc = response['current_condition'][0]['weatherDesc'][0]['value']
        
        return f"Il fait actuellement {temp}°C à {city} ({desc})."
    except Exception as e:
        return f"Oups, petite erreur technique avec l'API météo : {str(e)}"
    
@app.get("/chat")
def chat(message: str):
    message_clean = message.lower()
    vec = vectorizer.transform([message_clean])
    intent = model.predict(vec)[0]
    
    if intent in ("meteo_actuelle", "temperature"):
        # --- CORRECTION DE L'INDENTATION ICI ---
        ville = "Ottawa"  # Ville par défaut
        
        # Recherche de la ville après "à", "au" ou "a"
        pattern = r'(?:à|au|a)\s+([a-zA-Z\s\-]+)'
        match = re.search(pattern, message_clean)
        
        if match:
            # On récupère le premier mot trouvé après la préposition
            ville = match.group(1).split()[0].strip("?!.,;").title()
            
        data_meteo = get_real_weather(ville)
        return {"reponse": f"[Intention: {intent}] {data_meteo}"}
    
    elif intent == "salutation":
        return {"reponse": "Bonjour ! Je suis votre assistant météo. Comment puis-je vous aider ?"}
    
    elif intent == "conseil_vetement":
        return {"reponse": "Vu la météo, je vous conseille de prendre une petite veste !"}
    
    else:
        return {"reponse": f"J'ai bien compris votre demande ({intent}), mais je suis encore en apprentissage !"}
