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
    pattern = r"(?:\bà\s|\ben\s|\bdans\s|\bpour\s)([A-ZÀ-ÿ][a-zà-ÿ]+(?:\s[A-ZÀ-ÿ][a-zà-ÿ]+)*)"
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
@app.post("/ask")
async def ask_bot(query: UserQuery):
    msg = query.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message vide")

    # --- ÉTAPE 1 : CLASSIFICATION NLU ---
    msg_low = msg.lower()
    vec = vectorizer.transform([msg_low])
    prediction = model.predict(vec)[0]
    
    temps_intent = prediction[0]   # actuel / prévision / neutre
    sujet_intent = prediction[1]   # meteo_generale / pluie / vent / temperature / social

    # --- ÉTAPE 2 : GESTION DU SOCIAL ---
    if sujet_intent == "social":
        return {"reponse": "Bonjour ! Je suis prêt à vous donner la météo. Quelle ville vous intéresse ?"}

    # --- ÉTAPE 3 : EXTRACTION DE LA VILLE ---
    city = extract_city(msg)
    if not city:
        return {
            "intentions": {"temps": temps_intent, "sujet": sujet_intent},
            "reponse": "Je peux vous renseigner, mais j'ai besoin de savoir pour quelle ville."
        }

    # --- ÉTAPE 4 : RÉCUPÉRATION DES DONNÉES RÉELLES ---
    is_forecast = (temps_intent == "prévision")
    weather = get_weather_data(city, forecast=is_forecast)

    if not weather:
        return {"reponse": f"Désolé, je n'arrive pas à trouver les données pour {city}."}

    # --- ÉTAPE 5 : CONSTRUCTION DE LA RÉPONSE SELON LE SUJET ---
    prefix = "Demain à" if is_forecast else "Actuellement à"
    
    if sujet_intent == "temperature":
        if is_forecast:
            reponse = f"{prefix} {city}, il fera entre {weather['temp_min']}°C et {weather['temp_max']}°C."
        else:
            reponse = f"{prefix} {city}, il fait {weather['temp']}°C."
            
    elif sujet_intent == "vent":
        reponse = f"{prefix} {city}, le vent souffle à {weather['vent']} km/h."
        
    elif sujet_intent == "pluie":
        reponse = f"{prefix} {city}, les conditions sont : {weather['condition']}. Vérifiez les risques d'averses."
        
    else: # meteo_generale
        cond = weather['condition']
        t = weather['temp'] if not is_forecast else f"{weather['temp_min']}-{weather['temp_max']}°C"
        reponse = f"{prefix} {city}, le ciel est {cond} avec environ {t}."

    return {
        "analysis": {"temps": temps_intent, "sujet": sujet_intent, "ville": city},
        "reponse": reponse
    }

# 7. LANCEMENT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
