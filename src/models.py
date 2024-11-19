from pydantic import BaseModel

class HeaderIndices(BaseModel):
    indices: list[int]

class NewsSummary(BaseModel):
    num_summaries: int
    headers: list[str]
    summaries: list[str]

class FeedEntry(BaseModel):
    header: str
    content: str
    link: str

class Feed(BaseModel):
    website: str
    entries: list[FeedEntry]
