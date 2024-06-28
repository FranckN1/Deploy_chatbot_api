import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .functions import vocabs, clean_text, pred_class, bag_of_words, get_response

# Configuration de l'application FastAPI
app = FastAPI()


# Serve the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Définir le chemin absolu du répertoire parent
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Construire le chemin complet vers data.json
data_file_path = os.path.join(project_root, 'datas', 'data.json')

# Lire le fichier JSON
with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Création des listes
words, classes, doc_X, doc_y = vocabs(data)

# Modèle de données pour la requête
class MessageRequest(BaseModel):
    message: str

# Route pour le chatbot
@app.post("/chatbot/")
async def get_chatbot_response(request: MessageRequest):
    try:
        message = request.message
        intents = pred_class(message, words, classes)
        result = get_response(intents, data)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def get():
    with open(os.path.join(os.path.dirname(__file__), 'index.html')) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
