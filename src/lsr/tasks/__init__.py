"""Retrieval tasks. A task's name states its direction — see base.py."""

from lsr.tasks.base import Document, Modality, Query, Task, check_task
from lsr.tasks.clsr import ManifestTask, read_manifest

__all__ = [
    "Document",
    "ManifestTask",
    "Modality",
    "Query",
    "Task",
    "check_task",
    "read_manifest",
]
