"""
Database Initialization & Table Migration Script
Creates all relational database tables and seeds camera & parking slots.
"""

import hashlib
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.session import Base, engine, SessionLocal
from database.models import User, Camera, ParkingSlot
from database.crud import sync_slots_from_json, get_or_create_camera
from cv_engine.parking import DEFAULT_SLOTS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("database.init")


def init_database():
    print("=" * 68)
    print("        SMART PARKING AI - DATABASE INITIALIZATION & MIGRATION        ")
    print("=" * 68)

    # 1. Create all relational tables via SQLAlchemy metadata
    print("\n[Step 1] Creating relational database tables...")
    Base.metadata.create_all(bind=engine)
    print(f"  [OK] Tables created successfully using engine: {engine.url}")

    db = SessionLocal()
    try:
        # 2. Seed Default Administrator Account
        print("\n[Step 2] Seeding initial administrator account...")
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            # Simple SHA256 seed for phase 8 baseline (JWT & bcrypt in Phase 25)
            hashed = hashlib.sha256("admin123".encode()).hexdigest()
            admin = User(
                username="admin",
                email="admin@smartparking.ai",
                hashed_password=hashed,
                role="admin"
            )
            db.add(admin)
            db.commit()
            print("  [OK] Created default admin account: admin / admin123")
        else:
            print("  [OK] Admin account already exists.")

        # 3. Synchronize Parking Slots from JSON
        print("\n[Step 3] Synchronizing parking slots from parking_slots.json...")
        if DEFAULT_SLOTS_FILE.is_file():
            slots = sync_slots_from_json(db, DEFAULT_SLOTS_FILE)
            print(f"  [OK] Successfully synchronized {len(slots)} slots into database!")
            for s in slots:
                print(f"       -> Slot {s.slot_code} [{s.current_status}] (Camera: CAM01)")
        else:
            print(f"  [WARN] Slots configuration file not found at: {DEFAULT_SLOTS_FILE}")

        print("\n" + "=" * 68)
        print(">> DATABASE INITIALIZATION & SEEDING COMPLETED SUCCESSFULLY!")
        print("=" * 68)

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
