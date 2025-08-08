"""Database setup and connection management"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..utils.config import get_database_path
from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
engine: Engine = None
SessionLocal: sessionmaker = None


def init_database(database_url: str = None) -> None:
    """Initialize the database connection and create tables"""
    global engine, SessionLocal
    
    if database_url is None:
        database_url = get_database_path()
    
    logger.info(f"Initializing database: {database_url}")
    
    # Configure engine based on database type
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            echo=False,
        )
        
        # Enable WAL mode for SQLite for better concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
    else:
        engine = create_engine(database_url, echo=False)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def get_engine() -> Engine:
    """Get the database engine"""
    if engine is None:
        init_database()
    return engine


def get_session_factory() -> sessionmaker:
    """Get the session factory"""
    if SessionLocal is None:
        init_database()
    return SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup"""
    if SessionLocal is None:
        init_database()
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def create_tables() -> None:
    """Create all database tables"""
    if engine is None:
        init_database()
    Base.metadata.create_all(bind=engine)
    logger.info("All tables created successfully")


def drop_tables() -> None:
    """Drop all database tables (use with caution!)"""
    if engine is None:
        init_database()
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")


def reset_database() -> None:
    """Reset the database by dropping and recreating all tables"""
    logger.warning("Resetting database - all data will be lost!")
    drop_tables()
    create_tables()
    logger.info("Database reset completed")


class DatabaseManager:
    """Database manager for handling connections and sessions"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or get_database_path()
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._init_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            self._init_engine()
        return self._session_factory
    
    def _init_engine(self) -> None:
        """Initialize the database engine"""
        logger.info(f"Creating database engine: {self.database_url}")
        
        if self.database_url.startswith("sqlite"):
            self._engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            self._engine = create_engine(self.database_url, echo=False)
        
        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )
        
        # Create tables
        Base.metadata.create_all(bind=self._engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Tables created successfully")
    
    def drop_tables(self) -> None:
        """Drop all database tables"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Tables dropped")
    
    def reset(self) -> None:
        """Reset the database"""
        logger.warning("Resetting database")
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset completed")


# Global database manager instance
db_manager = DatabaseManager()