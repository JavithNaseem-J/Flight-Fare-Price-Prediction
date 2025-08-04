from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import pandas as pd
import json
from src.FareFinder.pipeline.predictionpipeline import PredictionPipeline

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
templates = Jinja2Templates(directory=template_dir)


static_dir = os.path.join(current_dir, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/api/placeholder/{width}/{height}")
async def placeholder_image(width: int, height: int):
    return JSONResponse(content={"width": width, "height": height})

@app.get("/")
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    Airline: str = Form(...),
    Source: str = Form(...),
    Destination: str = Form(...),
    Stopovers: str = Form(...),
    Class: str = Form(...),
    Booking_Source: str = Form(None, alias="Booking Source"),
    Days_Before_Departure: int = Form(None, alias="Days Before Departure"),
    Departure_Time: str = Form(None, alias="Departure Time"),
    Arrival_Time: str = Form(None, alias="Arrival Time")
):
    try:
        if Source == Destination:
            return JSONResponse(content={"error": "Source and destination cities cannot be the same"})

        if Departure_Time == Arrival_Time:
            return JSONResponse(content={"error": "Departure time and arrival time cannot be the same"})


        data = {
            "Airline": Airline,
            "Source": Source,
            "Destination": Destination,
            "Stopovers": Stopovers,
            "Class": Class,
            "Booking Source": Booking_Source,
            "Days Before Departure": Days_Before_Departure,
            "Departure Time": Departure_Time,
            "Arrival Time": Arrival_Time
        }


        input_df = pd.DataFrame({
            'Airline': [Airline],
            'Source': [Source],
            'Destination': [Destination],
            'Stopovers': [Stopovers],
            'Class': [Class],
            'Booking Source': [Booking_Source],
            'Days Before Departure': [Days_Before_Departure],
            'Departure Time': [Departure_Time],
            'Arrival Time': [Arrival_Time]
        })

        print(f"Input DataFrame columns: {input_df.columns.tolist()}")
        print(f"Input DataFrame values: {input_df.iloc[0].tolist()}")

        pipeline = PredictionPipeline()
        prediction = round(float(pipeline.predict(input_df)), 2)
        print(f"Prediction result: {prediction}")

        # Encode data for URL parameters
        encoded_data = json.dumps(data)
        
        # Return redirect instruction
        return JSONResponse(content={"redirect": f"/results?prediction={prediction}&data={encoded_data}"})

    except Exception as e:
        import traceback
        print(f"Error in prediction endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": f"Error during prediction: {str(e)}"}, status_code=500)

@app.get("/results")
async def show_results(request: Request, prediction: float = None, data: str = None):
    try:
        # Check if we have both required parameters
        if not prediction or not data:
            print("Missing required parameters for results page")
            return RedirectResponse(url="/")

        print(f"Received prediction: {prediction}")
        print(f"Received data: {data}")
        
        input_data = json.loads(data)
        
        print(f"Parsed input_data: {input_data}")
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "prediction": float(prediction),
            "input_data": input_data
        })
    except Exception as e:
        import traceback
        print(f"Error in results handler: {str(e)}")
        print(traceback.format_exc())
        return RedirectResponse(url="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)