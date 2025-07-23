import threading
from typing import Dict, Optional
from contextlib import contextmanager

# Global dictionary to store locks for each conference
_conference_locks: Dict[str, threading.Lock] = {}
# Lock to protect access to the _conference_locks dictionary
_locks_lock = threading.Lock()

@contextmanager
def conference_lock(conference_id: str, timeout: Optional[float] = None):
    """
    Context manager to acquire a lock for a specific conference.
    This ensures that only one thread can write to a conference at a time.
    
    Args:
        conference_id: The ID of the conference to lock
        timeout: Optional timeout in seconds. If None, waits indefinitely.
    
    Yields:
        bool: True if lock was acquired, False if timeout occurred
    """
    # Get or create the lock for this conference
    with _locks_lock:
        if conference_id not in _conference_locks:
            _conference_locks[conference_id] = threading.Lock()
        lock = _conference_locks[conference_id]
    
    # Try to acquire the lock
    acquired = lock.acquire(timeout=timeout)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release() 