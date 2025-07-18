from fastapi import FastAPI, Depends
import property_friends
from .middleware.auth import verify_api_key
from .routers import predictions

app = FastAPI(
    title="Property Friends API",
    description="A property valuation prediction API",
    version="0.1.0",
)

app.include_router(predictions.router)


@app.get("/")
async def root(api_key: str = Depends(verify_api_key)):
    return {"message": "Property Friends API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
