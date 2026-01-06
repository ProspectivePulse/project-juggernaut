import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    # Base Directory (project-juggernaut root)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Network Settings
    ROBOT_IP = os.getenv("ROBOT_IP", "192.168.1.9")  # Default fallback
    ROBOT_PORT = int(os.getenv("ROBOT_PORT", 5000))

    # Model Settings
    # Default to a generic name in the models folder
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "dqn_robot_latest.zip"))

    @classmethod
    def validate(cls):
        """Simple check to ensure critical vars are set."""
        if not cls.ROBOT_IP:
            raise ValueError("ROBOT_IP is not set in environment variables.")
        print(f"✅ Config Validated. Target: {cls.ROBOT_IP}:{cls.ROBOT_PORT}")


if __name__ == "__main__":
    Config.validate()