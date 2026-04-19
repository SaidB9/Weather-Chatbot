from fastapi import FastAPI
import joblib
import requests
import re

app = FastAPI()

# 1. Charger le modèle et le vectoriseur
model = joblib.load('model_meteo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_real_weather(city="Montreal"):
    """Cherche la vraie température sur internet"""
    try:
        # On demande le format JSON et le français
        url = f"http://wttr.in/{city}?format=j1&lang=fr"
        response = requests.get(url).json()
        
        # Extraction sécurisée des données
        temp = response['current_condition'][0]['temp_C']
        
        # On récupère la description météo (qui sera en FR grâce à l'URL)
        desc = response['current_condition'][0]['weatherDesc'][0]['value']
        
        return f"Il fait actuellement {temp}°C à {city} ({desc})."
    except Exception as e:
        # On affiche l'erreur précise pour le débuggage si besoin
        return f"Oups, petite erreur technique avec l'API météo : {str(e)}"
    
@app.get("/chat")
def chat(message: str):
    message_clean = message.lower()
    vec = vectorizer.transform([message_clean])
    intent = model.predict(vec)[0]
    
    if intent in ("meteo_actuelle", "temperature"):
    ville = "Ottawa"  # Par défaut pour Ottawa, ON
    
    # Pattern : après " à ", " au " ou " a " → mots jusqu'à ponctuation ou fin
    pattern = r'(?:à|au|a)\s+([A-Z][a-zA-Z\s]*(?:[A-Z][a-zA-Z\s]*){0,2})[\s\?\!\.,;]?'
    match = re.search(pattern, message_clean, re.IGNORECASE)
    
    if match:
        ville = match.group(1).strip().title()  # Nettoie et capitalise
            
        data_meteo = get_real_weather(ville)
        return {"reponse": f"[Intention: {intent}] {data_meteo}"}
    
    elif intent == "salutation":
        return {"reponse": "Bonjour ! Je suis votre assistant météo. Comment puis-je vous aider ?"}
    
    elif intent == "conseil_vetement":
        return {"reponse": "Vu la météo, je vous conseille de prendre une petite veste !"}
    
    else:
        return {"reponse": f"J'ai bien compris votre demande ({intent}), mais je suis encore en apprentissage pour cette fonction !"}

