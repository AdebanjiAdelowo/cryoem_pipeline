"""FastAPI application entry point for the cryoEM pipeline."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="CryoEM Pipeline API",
    description=(
        "Simulate synthetic cryoEM micrographs and particle stacks, "
        "then process them with CTF correction, particle picking, and "
        "class averaging."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
