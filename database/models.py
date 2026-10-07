"""
SQLAlchemy 2.0 Relational Database Models
Defines complete enterprise schema for Users, Cameras, Parking Zones, Slots,
Vehicles, Events, Occupancy Snapshots, ML Predictions, and System Alerts.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .session import Base


def utc_now():
    """Returns timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """System operator / administrator credentials."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="operator")  # admin, operator, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)


class Camera(Base):
    """Video ingestion sensors monitoring parking facilities."""
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_code = Column(String(32), unique=True, nullable=False, index=True)  # e.g., CAM01
    name = Column(String(100), nullable=False)
    stream_url = Column(String(255), nullable=False)
    resolution = Column(String(32), default="1280x720")
    target_fps = Column(Integer, default=30)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    zones = relationship("ParkingZone", back_populates="camera", cascade="all, delete-orphan")
    slots = relationship("ParkingSlot", back_populates="camera", cascade="all, delete-orphan")
    events = relationship("ParkingEvent", back_populates="camera")
    alerts = relationship("SystemAlert", back_populates="camera")


class ParkingZone(Base):
    """Logical grouping of slots (e.g. Zone A, Level 1)."""
    __tablename__ = "parking_zones"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=False, index=True)
    zone_code = Column(String(32), nullable=False, index=True)  # e.g., ZONE_A
    name = Column(String(100), nullable=False)
    boundary_polygon = Column(JSON, nullable=True)  # Optional outer boundary polygon
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    camera = relationship("Camera", back_populates="zones")
    slots = relationship("ParkingSlot", back_populates="zone", cascade="all, delete-orphan")


class ParkingSlot(Base):
    """Individual parking bay with geometric coordinates and real-time state."""
    __tablename__ = "parking_slots"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=False, index=True)
    zone_id = Column(String(36), ForeignKey("parking_zones.id"), nullable=True, index=True)
    slot_code = Column(String(32), nullable=False, index=True)  # e.g., P01, P02
    polygon_coordinates = Column(JSON, nullable=False)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    current_status = Column(String(32), default="AVAILABLE", index=True)  # AVAILABLE, OCCUPIED, RESERVED
    assigned_vehicle_id = Column(Integer, nullable=True)
    last_status_change = Column(DateTime(timezone=True), default=utc_now)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    camera = relationship("Camera", back_populates="slots")
    zone = relationship("ParkingZone", back_populates="slots")
    events = relationship("ParkingEvent", back_populates="slot")
    history = relationship("OccupancyHistory", back_populates="slot", cascade="all, delete-orphan")


class Vehicle(Base):
    """Tracked vehicle identities associated across camera frames."""
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=False, index=True)  # ByteTrack integer ID
    vehicle_type = Column(String(32), default="car", index=True)  # car, motorcycle, bus, truck
    first_detected_at = Column(DateTime(timezone=True), default=utc_now)
    last_detected_at = Column(DateTime(timezone=True), default=utc_now)
    total_dwell_sec = Column(Float, default=0.0)

    # Relationships
    events = relationship("ParkingEvent", back_populates="vehicle")


class ParkingEvent(Base):
    """
    Immutable audit ledger of every parking state change and tripwire crossing.
    High-frequency query table with composite indexes.
    """
    __tablename__ = "parking_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=False, index=True)
    slot_id = Column(String(36), ForeignKey("parking_slots.id"), nullable=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)  # SLOT_OCCUPIED, SLOT_AVAILABLE, etc.
    confidence = Column(Float, default=1.0)
    timestamp = Column(DateTime(timezone=True), default=utc_now, index=True)
    extra_metadata = Column(JSON, nullable=True)

    # Relationships
    camera = relationship("Camera", back_populates="events")
    slot = relationship("ParkingSlot", back_populates="events")
    vehicle = relationship("Vehicle", back_populates="events")


class OccupancyHistory(Base):
    """
    Time-series periodic snapshots (e.g., recorded every 1–5 minutes)
    used for historical trend analytics and ML feature training.
    """
    __tablename__ = "occupancy_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    slot_id = Column(String(36), ForeignKey("parking_slots.id"), nullable=False, index=True)
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=False, index=True)
    is_occupied = Column(Boolean, nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), default=utc_now, index=True)

    # Relationships
    slot = relationship("ParkingSlot", back_populates="history")


class ModelPrediction(Base):
    """Machine learning occupancy forecasts generated by XGBoost."""
    __tablename__ = "model_predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=False, index=True)
    target_time = Column(DateTime(timezone=True), nullable=False, index=True)
    horizon_minutes = Column(Integer, nullable=False)  # 30, 60, 90 mins ahead
    predicted_occupancy_rate = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    model_version = Column(String(64), default="xgboost_v1")
    created_at = Column(DateTime(timezone=True), default=utc_now)


class SystemAlert(Base):
    """System-level operational warnings and anomalies."""
    __tablename__ = "system_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True, index=True)
    alert_type = Column(String(64), nullable=False, index=True)  # CAMERA_OFFLINE, PARKING_FULL, etc.
    severity = Column(String(32), default="INFO", index=True)  # INFO, WARNING, CRITICAL
    message = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True)

    # Relationships
    camera = relationship("Camera", back_populates="alerts")


# Explicit Composite Indexes for High-Frequency Analytical Queries
Index("idx_events_camera_time", ParkingEvent.camera_id, ParkingEvent.timestamp.desc())
Index("idx_events_slot_time", ParkingEvent.slot_id, ParkingEvent.timestamp.desc())
Index("idx_events_vehicle_time", ParkingEvent.vehicle_id, ParkingEvent.timestamp.desc())
Index("idx_history_slot_time", OccupancyHistory.slot_id, OccupancyHistory.recorded_at.desc())
Index("idx_predictions_time_horizon", ModelPrediction.target_time, ModelPrediction.horizon_minutes)
Index("idx_alerts_severity_time", SystemAlert.severity, SystemAlert.created_at.desc())
