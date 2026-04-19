import joblib
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. CONFIGURATION DE L'APPLICATION
app = FastAPI(
    title="Météo Bot Intelligent",
    description="Assistant météo utilisant le Machine Learning pour comprendre les intentions."
)

# 2. CHARGEMENT DES MODÈLES NLU
# Assure-toi que ces fichiers sont dans le même dossier
try:
    model = joblib.load('model_multilabel.pkl')
    vectorizer = joblib.load('vectorizer_multilabel.pkl')
except Exception as e:
    print(f"CRITICAL: Impossible de charger les modèles .pkl : {e}")

# 3. MODÈLE DE DONNÉES (INPUT)
class UserQuery(BaseModel):
    message: str

# 4. FONCTION D'EXTRACTION DE LA VILLE (Regex)
def extract_city(text: str) -> Optional[str]:
    # Cherche un nom propre après "à", "en", "dans" ou "pour"
    # Cette version accepte "à", "a", "sur", "pour", "en" ou "dans"
    pattern = r"(?:\bà\s|\ba\s|\bsur\s|\bpour\s|\ben\s|\bdans\s)([A-ZÀ-ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-ÿ][a-zà-ÿ]+)*)"
    
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

# 5. SERVICE EXTERNE : APPEL À WTTR.IN
def get_weather_data(city: str, forecast: bool = False):
    """
    Récupère les données depuis wttr.in au format JSON.
    forecast=False -> Météo actuelle
    forecast=True  -> Prévisions pour demain
    """
    try:
        # On demande le format JSON (?format=j1)
        url = f"http://wttr.in/{city}?format=j1&lang=fr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if forecast:
            # On récupère les prévisions du jour suivant (index 1)
            f_data = data['weather'][1]
            return {
                "temp_max": f_data['maxtempC'],
                "temp_min": f_data['mintempC'],
                "condition": f_data['hourly'][4]['lang_fr'][0]['value'], # Midi
                "vent": f_data['hourly'][4]['windspeedKmph']
            }
        else:
            # Météo actuelle
            current = data['current_condition'][0]
            return {
                "temp": current['temp_C'],
                "condition": current['lang_fr'][0]['value'],
                "vent": current['windspeedKmph'],
                "humidite": current['humidity']
            }
    except Exception as e:
        return None

# 6. POINT D'ENTRÉE (ENDPOINT) PRINCIPAL
# 1. On change @app.post en @app.get
# 2. On s'assure que le chemin est bien "/chat" pour correspondre à ton URL
@app.get("/chat")
async def chat_endpoint(message: str = None): # Le message arrive directement via l'URL
    if not message:
        return {"error": "Aucun message reçu. Ajoutez ?message=votre_phrase à l'URL."}

    # --- ÉTAPE 1 : CLASSIFICATION NLU ---
    msg_low = message.lower().strip()

    # Règle prioritaire pour les salutations
    if msg_low in ["bonjour", "salut", "hello", "coucou"]:
        return {"reponse": "Bonjour ! Je suis votre assistant météo. Posez-moi une question sur le temps qu'il fait."}

    # Sinon, on utilise le modèle ML...
    vec = vectorizer.transform([msg_low])
    prediction = model.predict(vec)[0]
    
    temps_intent = prediction[0]   
    sujet_intent = prediction[1]   

    # --- ÉTAPE 2 : GESTION DU SOCIAL ---
    if sujet_intent == "social":
        return {"reponse": "Bonjour ! Je suis prêt à vous donner la météo. Quelle ville vous intéresse ?"}

    # --- ÉTAPE 3 : EXTRACTION DE LA VILLE ---
    city = extract_city(message)
    if not city:
        return {
            "intentions": {"temps": temps_intent, "sujet": sujet_intent},
            "reponse": f"J'ai détecté une demande pour {sujet_intent}, mais je n'ai pas trouvé la ville dans votre phrase."
        }

    # --- ÉTAPE 4 : APPEL À WTTR.IN ---
    is_forecast = (temps_intent == "prévision")
    weather = get_weather_data(city, forecast=is_forecast)

    if not weather:
        return {"reponse": f"Désolé, les données météo pour {city} sont indisponibles."}

    # --- ÉTAPE 5 : CONSTRUCTION DE LA RÉPONSE ---
    prefix = "Demain à" if is_forecast else "Actuellement à"
    if sujet_intent == "temperature":
        reponse = f"{prefix} {city}, il fera {weather.get('temp', weather.get('temp_max'))}°C."
    elif sujet_intent == "vent":
        reponse = f"{prefix} {city}, le vent est de {weather['vent']} km/h."
    else:
        reponse = f"{prefix} {city}, le ciel est {weather['condition']}."

    return {
        "analysis": {"temps": temps_intent, "sujet": sujet_intent, "ville": city},
        "reponse": reponse
    }

# 7. LANCEMENT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
