import pydantic
from pydantic import ConfigDict, validator
from typing_extensions import ClassVar, List, Literal, Union


class BaseModel(pydantic.BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

class ImageRequest(BaseModel):
    # input or list of inputs
    prompt: str
    # model to query
    model: str
    # num generation steps
    steps: Union[int, None] = 20
    # seed
    seed: Union[int, None] = None
    # number of results to return
    n: Union[int, None] = 1
    # pixel height
    height: Union[int, None] = 1024
    # pixel width
    width: Union[int, None] = 1024
    # negative prompt
    negative_prompt: Union[str, None] = None

    @validator('prompt')
    def check_not_empty(cls, v):
        if not v or v == '':
            raise ValueError('prompt cannot be empty or null')
        return v

    @validator("steps")
    def check_valid_steps(cls, v):
        if v <= 0:
            raise ValueError("steps must be greater than zero")
        if v > 150:
            raise ValueError("steps must be less than 150")
        return v
    
    @validator("seed")
    def check_valid_seed(cls, v):
        if (v is not None) and v < 0:
            raise ValueError("seed must be greater than zero")
        return v

    @validator("model")
    def check_valid_model(cls, v):
        if v != 'stabilityai/stable-diffusion-xl-base-1.0':
            raise ValueError("model is not found")
        return v

    @validator("n")
    def check_valid_n(cls, v):
        if (v < 1) or (v > 4):
            raise ValueError("invalid n, please in [1, 4]")
        return v

    @validator("height")
    def check_valid_height(cls, v):
        if (v < 512) or (v > 2048):
            raise ValueError("invalid height, please in [512, 2048]")
        return v

    @validator("width")
    def check_valid_width(cls, v):
        if (v < 512) or (v > 2048):
            raise ValueError("invalid width, please in [512, 2048]")
        return v
    
    def is_valid(self):
        self.check_not_empty(self.prompt)
        self.check_valid_steps(self.steps)
        self.check_valid_seed(self.seed)
        self.check_valid_model(self.model)
        self.check_valid_n(self.n)
        self.check_valid_height(self.height)
        self.check_valid_width(self.width)



class ImageChoicesData(BaseModel):
    # response index
    index: int
    # base64 image response
    b64_json: Union[str, None] = None
    # URL hosting image
    url: Union[str, None] = None

class ImageResponse(BaseModel):
    # job id
    id: Union[str, None] = None
    # query model
    model: Union[str, None] = None
    # object type
    object: Union[Literal["list"], None] = None
    # list of embedding choices
    data: Union[List[ImageChoicesData], None] = None
