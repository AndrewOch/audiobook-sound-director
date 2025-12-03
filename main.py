# from uvicorn import run

# if __name__ == "__main__":
#     # Run FastAPI app from the app package
#     run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    import os
    import uvicorn

    reload_enabled = os.environ.get("AUDIOBOOK_RELOAD") == "1"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
    )
