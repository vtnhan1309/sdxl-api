import http
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import config


class AuthMiddleware(BaseHTTPMiddleware):
    def dispatch(self, request: Request, call_next):
        # Check if the Authorization header exists
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise HTTPException(status_code=http.HTTPStatus.UNAUTHORIZED, 
                                detail="Authorization header missing")
        
        token = auth_header
        
        if not token or token != config.AUTH_TOKEN:  # Replace with your validation logic
            raise HTTPException(status_code=http.HTTPStatus.FORBIDDEN, 
                                detail="Invalid token")
        
        # Proceed with the request if valid
        response = call_next(request)
        return response
    
