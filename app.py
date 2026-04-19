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
    # On ajoute des espaces autour des prépositions pour ne pas matcher l'intérieur d'un mot
    # On cherche " à ", " au ", ou " a "
    pattern = r'\s(?:à|au|a)\s+([a-zA-Z\u00C0-\u017F\s\-]+)'
    
    # On ajoute un espace au début du message pour que la regex marche même si la ville est au début
    match = re.search(pattern, " " + message.lower(), re.IGNORECASE)
    
    if match:
        extraction = match.group(1).strip()
        mots = extraction.split()
        
        # Liste d'exclusion pour ne pas prendre de mots parasites
        mots_interdits = ["la", "le", "les", "une", "un", "temperature", "température", "météo", "aujourd'hui", "aujourdhui"]
        
        for mot in mots:
            mot_propre = mot.strip("?!.,;")
            if mot_propre.lower() not in mots_interdits and len(mot_propre) > 1:
                return mot_propre.title()
                
    return "Ottawa" # Ville par défaut si rien n'est trouvé

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
