import dotenv
import os
from pydantic import BaseModel
from typing import List

from src.llm import NewsSummary

class News(BaseModel):
    num_news: int
    links: List[str]
    content: NewsSummary

class FeedEntry(BaseModel):
    header: str
    content: str
    link: str

class Feed(BaseModel):
    website: str
    entries: List[FeedEntry]


class EntrySummary(BaseModel):
    header_summary: str
    content_summary: str
    link: str

class FeedSummary(BaseModel):
    website: str
    summary_entries: List[EntrySummary]




# TODO: cronjob should run this main.py


# TODO: use scraper to scrape rss feeds from atv, and other news feed

# TODO: prepare groq requets to send the feed and get summary (prepare prompt)

# TODO: retrieve prompts from groq and embed them into an email template

# TODO: send the email to users subscribed?


