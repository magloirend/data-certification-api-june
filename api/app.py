import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness,danceability,duration_ms,energy,\
    explicit,id,instrumentalness,key,liveness,loudness,\
    mode,name,release_date,speechiness,tempo,valence,artist):


    x = dict(
      acousticness= float(acousticness),
      danceability= float(danceability),
      duration_ms= int(duration_ms),
      energy= float(energy),
      explicit= int(explicit),
      id= str(id),
      instrumentalness=float(instrumentalness),
      key=int(key),
      liveness=float(liveness),
      loudness=float(loudness),
      mode=int(mode),
      name=str(name),
      release_date=str(release_date),
      speechiness=float(speechiness),
      tempo=float(tempo),
      valence=float(valence),
      artist=str(artist)
    )
    # data_dict = pd.DataFrame.from_records(data)
    data = pd.DataFrame.from_records([x])
    # print(data_dict)


    pipeline = joblib.load('./model.joblib')
    y_pred = float(pipeline.predict(data)[0])

    return { 'popularity' : y_pred,
    "artist": f"{artist}",
    "name": f"{name}" }
