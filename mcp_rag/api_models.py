"""Pydantic request/response models for the code-rag REST API."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    path: str
    summary: str
    score: float


class UnitSummary(BaseModel):
    path: str
    summary: str


class UnitDetail(BaseModel):
    path: str
    content: str
    summary: str


class FileEntry(BaseModel):
    repo: str
    path: str
    indexed_at: str


class RepoEntry(BaseModel):
    name: str
    root: str
    added_at: str
    description: str


class IndexStatusRepo(BaseModel):
    repo: str
    root: str
    file_count: int
    unit_count: int
    last_indexed_at: str | None = None


class IndexStatus(BaseModel):
    repos: list[IndexStatusRepo]
    total_units: int
    embed_count: int


class BrowseNode(BaseModel):
    type: str
    name: str
    path: str
    summary: str
    has_children: bool
    unit_type: str | None = None


class DirEntry(BaseModel):
    name: str
    path: str


class LsResponse(BaseModel):
    path: str
    parent: str | None
    is_git: bool
    dirs: list[DirEntry]


class FetchRequest(BaseModel):
    paths: list[str]


class IndexRequest(BaseModel):
    paths: list[str]
    reindex: bool = False


class JobStatus(BaseModel):
    running: bool
    last_result: str | None = None
    last_finished_at: str | None = None
    queue: list[str] = []
