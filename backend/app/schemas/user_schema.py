from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

class LoginRequest(BaseModel):
     username: str = Field(..., min_length=3, max_length=50)
     password: str = Field(..., min_length=8)
class LoginResponse(BaseModel):
    message: str
    user_id: int
    username: str
