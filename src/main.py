import dotenv
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime

from scraper import RSSParser
from llm import NewsSummary, GroqClient
from email_client import EmailClient
from models import Feed, FeedEntry, NewsSummary

RSS_URL = "https://www.ntv.com.tr/gundem.rss"

class News(BaseModel):
    num_news: int
    links: List[str]
    content: NewsSummary



class EntrySummary(BaseModel):
    header_summary: str
    content_summary: str
    link: str

class FeedSummary(BaseModel):
    website: str
    summary_entries: List[EntrySummary]


# TODO: cronjob should run this main.py


# TODO: use scraper to scrape rss feeds from atv, and other news feed

parser = RSSParser(RSS_URL)
entries = parser.get_entries_by_date(datetime.now())
feed = parser.serialize_to_feed(entries)


client = GroqClient()
# TODO: define preference in config.yml
model_preference_list = ['llama-3.1-70b-versatile', 'mixtral-8x7b-32768', 'llama3-70b-8192']
# model_preference_list = ['llama3-groq-70b-8192-tool-use-preview', 'llama3-8b-8192 gemma2-9b-it', 'gemma2-9b-it', 'llama3-70b-8192']
# model_preference_list = ['llama3-8b-8192 gemma2-9b-it', 'gemma2-9b-it', 'llama3-70b-8192']
# model_preference_list = ['gemma2-9b-it', 'llama3-70b-8192']
active_models = client.get_active_models()
print(f'-> found active models: {active_models}')

model_name = None
for name in model_preference_list:
    if name in active_models:
        print(f'-> using {name}')
        model_name = name
        break

if model_name is None:
    raise ValueError('no preferred model is available')

# choose the most important headers
header_indices: list[int] = client.choose_headers(feed, count=5, model_id=model_name)
filtered_entries: list[FeedEntry] = []
for idx in header_indices:
    filtered_entries.append(feed.entries[idx])

filtered_feed: Feed = Feed(website=feed.website, entries=filtered_entries)

summary: NewsSummary = client.summarize_feed(filtered_feed, model_name)

# send email to subscribers
email_client = EmailClient()
email_client.send_email_to_all(summary)
