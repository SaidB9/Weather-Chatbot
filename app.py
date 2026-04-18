from fastapi import FastAPI
import joblib
import requests

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
    # Transformation et prédiction
    vec = vectorizer.transform([message.lower()])
    intent = model.predict(vec)[0]
    
    # Logique de réponse selon l'intention
    if intent == "temperature" or intent == "meteo_actuelle":
        # Note : Dans un vrai projet Azure, on utiliserait le NLP pour extraire la ville.
        # Ici, on simule pour démontrer la connexion API.
        data_meteo = get_real_weather("Paris") # Ville exemple
        return {"reponse": f"[Intention: {intent}] {data_meteo}"}
    
    elif intent == "salutation":
        return {"reponse": "Bonjour ! Je suis votre assistant météo. Comment puis-je vous aider ?"}
    
    elif intent == "conseil_vetement":
        return {"reponse": "Vu la météo, je vous conseille de prendre une petite veste !"}
    
    else:
        return {"reponse": f"J'ai bien compris votre demande ({intent}), mais je suis encore en apprentissage pour cette fonction !"}

# Pour tester : http://127.0.0.1:8000/chat?message=Quel temps fait-il ?
