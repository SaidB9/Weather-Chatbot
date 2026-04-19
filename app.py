from fastapi import FastAPI
import joblib
import requests
import re

app = FastAPI()

# 1. Charger le modèle et le vectoriseur
model = joblib.load('model_meteo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_real_weather(city="Ottawa"):
    """Cherche la météo actuelle (Direct)"""
    try:
        url = f"http://wttr.in/{city}?format=j1&lang=fr"
        response = requests.get(url).json()
        temp = response['current_condition'][0]['temp_C']
        desc = response['current_condition'][0]['weatherDesc'][0]['value']
        return f"Il fait actuellement {temp}°C à {city} ({desc})."
    except Exception as e:
        return f"Erreur météo actuelle : {str(e)}"

def get_forecast(city="Ottawa"):
    """Cherche les prévisions pour demain"""
    try:
        url = f"http://wttr.in/{city}?format=j1&lang=fr"
        response = requests.get(url).json()
        
        # weather[1] correspond à demain dans l'API wttr.in
        forecast_tomorrow = response['weather'][1]
        date = forecast_tomorrow['date']
        max_t = forecast_tomorrow['maxtempC']
        min_t = forecast_tomorrow['mintempC']
        # On prend la description météo à midi (index 4 de 'hourly')
        desc = forecast_tomorrow['hourly'][4]['weatherDesc'][0]['value']
        
        return f"Demain le {date} à {city}, il fera entre {min_t}°C et {max_t}°C ({desc})."
    except Exception as e:
        return f"Erreur prévisions : {str(e)}"

def extraire_ville(message):
    """Extraction améliorée pour éviter de confondre l'intention et la ville"""
    # On cherche spécifiquement ce qui suit les prépositions de lieu
    # en ignorant les mots fonctionnels comme "la" ou "le"
    pattern = r'(?:à|au|a)\s+([a-zA-Z\u00C0-\u017F\s\-]+)' 
    match = re.search(pattern, message, re.IGNORECASE)
    
    if match:
        # On nettoie pour ne prendre que le premier groupe de mots significatifs
        brut = match.group(1).strip()
        # On divise par les espaces et on prend ce qui reste après avoir 
        # éliminé les mots vides si nécessaire, ici on prend le dernier bloc 
        # car la ville est souvent à la fin dans "température à Casablanca"
        mots = brut.split()
        if mots:
            # On prend le premier mot trouvé après la préposition
            ville_potentielle = mots[0].strip("?!.,;")
            return ville_potentielle.title()
            
    return "Ottawa" # Ville par défaut

@app.get("/chat")
def chat(message: str):
    message_clean = message.lower()
    vec = vectorizer.transform([message_clean])
    intent = model.predict(vec)[0]
    
    # 1. Météo Actuelle
    if intent in ("meteo_actuelle", "temperature"):
        ville = extraire_ville(message_clean)
        data = get_real_weather(ville)
        return {"reponse": f"[Intention: {intent}] {data}"}
    
    # 2. Prévisions (NOUVEAU)
    elif intent == "previsions":
        ville = extraire_ville(message_clean)
        data = get_forecast(ville)
        return {"reponse": f"[Intention: {intent}] {data}"}
    
    # 3. Salutations
    elif intent == "salutation":
        return {"reponse": "Bonjour ! Je suis votre assistant météo. Comment puis-je vous aider ?"}
    
    # 4. Conseils
    elif intent == "conseil_vetement":
        return {"reponse": "Vu la météo, je vous conseille de prendre une petite veste !"}
    
    # 5. Fallback
    else:
        return {"reponse": f"J'ai compris votre demande ({intent}), mais je ne sais pas encore traiter ce détail !"}
