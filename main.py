from api.config import create_app
from api.routes import chat_routes, health_routes
import uvicorn

app = create_app()

app.include_router(chat_routes.router)
app.include_router(health_routes.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
