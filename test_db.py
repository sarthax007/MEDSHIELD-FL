import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def test_connection():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL_SYNC")
    if not db_url:
        print("DATABASE_URL_SYNC not found in .env")
        return

    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print(f"Connection successful! Query returned: {result.scalar()}")
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    test_connection()
