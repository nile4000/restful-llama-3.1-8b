# Each worker takes up roughly 20GB RAM to load the model
gunicorn -b 0.0.0.0:7860 app:app --workers 1 -k uvicorn.workers.UvicornWorker --timeout 600