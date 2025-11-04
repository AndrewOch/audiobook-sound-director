from uvicorn import run

if __name__ == "__main__":
    # Run FastAPI app from the app package
    run("app.main:app", host="0.0.0.0", port=8000, reload=True)
