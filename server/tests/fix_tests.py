import glob

for f in glob.glob("server/tests/test_*.py"):
    with open(f, "r") as file:
        content = file.read()

    if "StaticPool" not in content:
        content = content.replace(
            "from sqlalchemy import create_engine",
            "from sqlalchemy import create_engine\nfrom sqlalchemy.pool import StaticPool",
        )
        content = content.replace(
            'connect_args={"check_same_thread": False})',
            'connect_args={"check_same_thread": False}, poolclass=StaticPool)',
        )

        # also move create_all inside setup_db if it's outside
        content = content.replace("models.Base.metadata.create_all(bind=engine)\n", "")
        content = content.replace("Base.metadata.create_all(bind=engine)\n", "")
        content = content.replace(
            "def setup_db():\n",
            "def setup_db():\n    from app.db.base import Base\n    Base.metadata.create_all(bind=engine)\n",
        )

        with open(f, "w") as file:
            file.write(content)
