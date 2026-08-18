import os
import json
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, Float, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class RoundMetric(Base):
    __tablename__ = 'round_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, index=True)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    participating_clients = Column(String, nullable=True)  # JSON string

class GlobalModel(Base):
    __tablename__ = 'global_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, unique=True, index=True)
    model_weights = Column(LargeBinary, nullable=False)

db_url = os.getenv("DATABASE_URL_SYNC", "sqlite:///./fl_metrics.db")
engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def save_metrics(round_number: int, accuracy: float, loss: float, clients: list):
    session = SessionLocal()
    try:
        clients_json = json.dumps(clients)
        metric = RoundMetric(
            round_number=round_number, 
            accuracy=accuracy, 
            loss=loss, 
            participating_clients=clients_json
        )
        session.add(metric)
        session.commit()
    finally:
        session.close()

def save_global_model(round_number: int, model_weights: bytes):
    session = SessionLocal()
    try:
        model = GlobalModel(round_number=round_number, model_weights=model_weights)
        session.add(model)
        session.commit()
    finally:
        session.close()

def load_global_model(round_number: int) -> Optional[bytes]:
    session = SessionLocal()
    try:
        model = session.query(GlobalModel).filter_by(round_number=round_number).first()
        return model.model_weights if model else None  # type: ignore
    finally:
        session.close()
