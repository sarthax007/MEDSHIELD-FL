from typing import List, Union
from pydantic import AnyHttpUrl, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    FL_SERVER_HOST: str = "localhost"
    FL_SERVER_PORT: int = 8080
    FL_NUM_ROUNDS: int = 10
    FL_MIN_CLIENTS: int = 2
    
    HE_POLY_MOD_DEGREE: int = 8192
    HE_COEFF_MOD_BIT_SIZES: str = "60,40,40,60"
    HE_SCALE: float = 2**40
    HE_SECRET_CONTEXT_PATH: str = "./keys/secret.ctx"
    HE_PUBLIC_CONTEXT_PATH: str = "./keys/public.ctx"
    
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: Union[str, List[str]] = []

    DATA_RAW_DIR: str = "./data/raw"
    DATA_PROCESSED_DIR: str = "./data/processed"
    DATA_PARTITIONS_DIR: str = "./data/partitions"
    DATA_CHECKPOINTS_DIR: str = "./data/checkpoints"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @field_validator("HE_SCALE", mode="before")
    def parse_he_scale(cls, v: Union[str, float]) -> float:
        if isinstance(v, str) and "**" in v:
            base, exp = v.split("**")
            return float(int(base) ** int(exp))
        return float(v)

    @property
    def parsed_he_coeff_mod_bit_sizes(self) -> List[int]:
        return [int(x.strip()) for x in self.HE_COEFF_MOD_BIT_SIZES.split(',')]

settings = Settings()
