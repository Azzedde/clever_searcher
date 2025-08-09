"""Operation logging for complete execution tracking"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid

from ..utils.config import get_data_dir

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for operation entries"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Individual log entry within an operation"""
    timestamp: datetime
    level: LogLevel
    component: str
    operation: str
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component,
            "operation": self.operation,
            "message": self.message,
        }
        
        if self.data:
            result["data"] = self.data
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.error:
            result["error"] = self.error
            
        return result


@dataclass
class OperationLog:
    """Complete log for a single discovery operation"""
    operation_id: str
    user_query: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # running, completed, failed
    entries: List[LogEntry]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operation_id": self.operation_id,
            "user_query": self.user_query,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "entries": [entry.to_dict() for entry in self.entries],
            "metadata": self.metadata,
        }


class OperationLogger:
    """Logger for complete operation tracking"""
    
    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or (get_data_dir() / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_operation: Optional[OperationLog] = None
        self._operation_start_time: Optional[datetime] = None
    
    def start_operation(
        self, 
        user_query: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new operation log"""
        operation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        self.current_operation = OperationLog(
            operation_id=operation_id,
            user_query=user_query,
            started_at=timestamp,
            completed_at=None,
            status="running",
            entries=[],
            metadata=metadata or {},
        )
        
        self._operation_start_time = timestamp
        
        # Log the start
        self.log(
            level=LogLevel.INFO,
            component="operation_logger",
            operation="start_operation",
            message=f"Started operation for query: {user_query}",
            data={"operation_id": operation_id, "metadata": metadata}
        )
        
        return operation_id
    
    def log(
        self,
        level: LogLevel,
        component: str,
        operation: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Add a log entry to the current operation"""
        if not self.current_operation:
            logger.warning("No active operation to log to")
            return
        
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            component=component,
            operation=operation,
            message=message,
            data=data,
            duration_ms=duration_ms,
            error=error,
        )
        
        self.current_operation.entries.append(entry)
        
        # Also log to standard logger
        std_level = getattr(logging, level.value.upper())
        logger.log(std_level, f"[{component}:{operation}] {message}")
    
    def log_step_start(self, component: str, operation: str, message: str, data: Optional[Dict[str, Any]] = None) -> datetime:
        """Log the start of a step and return timestamp for duration calculation"""
        start_time = datetime.utcnow()
        self.log(
            level=LogLevel.INFO,
            component=component,
            operation=operation,
            message=f"Starting: {message}",
            data=data
        )
        return start_time
    
    def log_step_end(
        self, 
        component: str, 
        operation: str, 
        message: str, 
        start_time: datetime,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log the end of a step with duration"""
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        level = LogLevel.ERROR if error else LogLevel.INFO
        final_message = f"Completed: {message}" if not error else f"Failed: {message}"
        
        self.log(
            level=level,
            component=component,
            operation=operation,
            message=final_message,
            data=data,
            duration_ms=duration_ms,
            error=error
        )
    
    def complete_operation(self, status: str = "completed", final_data: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Complete the current operation and save to file"""
        if not self.current_operation:
            logger.warning("No active operation to complete")
            return None
        
        self.current_operation.completed_at = datetime.utcnow()
        self.current_operation.status = status
        
        if final_data:
            self.current_operation.metadata.update(final_data)
        
        # Calculate total duration
        if self._operation_start_time:
            total_duration = (self.current_operation.completed_at - self._operation_start_time).total_seconds()
            self.current_operation.metadata["total_duration_seconds"] = total_duration
        
        # Log completion
        self.log(
            level=LogLevel.INFO,
            component="operation_logger",
            operation="complete_operation",
            message=f"Operation completed with status: {status}",
            data={"final_metadata": final_data}
        )
        
        # Save to file
        log_file = self._save_operation_log()
        
        # Clear current operation
        self.current_operation = None
        self._operation_start_time = None
        
        return log_file
    
    def fail_operation(self, error_message: str, error_data: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Mark operation as failed and save"""
        if not self.current_operation:
            logger.warning("No active operation to fail")
            return None
        
        self.log(
            level=LogLevel.ERROR,
            component="operation_logger",
            operation="fail_operation",
            message="Operation failed",
            data=error_data,
            error=error_message
        )
        
        return self.complete_operation(status="failed", final_data={"error": error_message})
    
    def _save_operation_log(self) -> Path:
        """Save the current operation log to file"""
        if not self.current_operation:
            raise ValueError("No current operation to save")
        
        # Generate filename with timestamp and operation ID
        timestamp_str = self.current_operation.started_at.strftime("%Y%m%d_%H%M%S")
        filename = f"operation_{timestamp_str}_{self.current_operation.operation_id[:8]}.json"
        log_file = self.logs_dir / filename
        
        # Save as JSON
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_operation.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Operation log saved to: {log_file}")
        return log_file
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operation logs"""
        log_files = sorted(
            self.logs_dir.glob("operation_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        logs = []
        for log_file in log_files[:limit]:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    logs.append(log_data)
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
        
        return logs
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get statistics about logged operations"""
        log_files = list(self.logs_dir.glob("operation_*.json"))
        
        if not log_files:
            return {"total_operations": 0}
        
        stats = {
            "total_operations": len(log_files),
            "completed": 0,
            "failed": 0,
            "avg_duration_seconds": 0,
            "total_entries": 0,
        }
        
        total_duration = 0
        total_entries = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    
                    if log_data.get("status") == "completed":
                        stats["completed"] += 1
                    elif log_data.get("status") == "failed":
                        stats["failed"] += 1
                    
                    duration = log_data.get("metadata", {}).get("total_duration_seconds", 0)
                    total_duration += duration
                    
                    total_entries += len(log_data.get("entries", []))
                    
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
        
        if stats["total_operations"] > 0:
            stats["avg_duration_seconds"] = total_duration / stats["total_operations"]
            stats["avg_entries_per_operation"] = total_entries / stats["total_operations"]
        
        stats["total_entries"] = total_entries
        
        return stats


# Global operation logger instance
operation_logger = OperationLogger()