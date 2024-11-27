import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List

import feedparser
import requests
from bs4 import BeautifulSoup

from models import Feed, FeedEntry


class BaseRssParser(ABC):
    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.rss_feed = feedparser.parse(rss_url)

    def _html_strip(self, html_str: str) -> str:
        return re.sub(r"<.*?>", "", html_str)

    def _get_html(self, url: str) -> str:
        response = requests.get(url)
        html = BeautifulSoup(response.text, "html.parser")
        return html

    def get_entries_by_date(self, date: datetime) -> List[Any]:
        entries = []
        for entry in self.rss_feed.entries:
            # published_parsed return 9 tuple in UTC
            # we get all entries that are published on the same day
            if (
                entry.published_parsed[0] == date.year
                and entry.published_parsed[1] == date.month
                and entry.published_parsed[2] == date.day
            ):
                entries.append(entry)
        return entries

    @abstractmethod
    def create_feed_entry_from_feedparser(self, entry: List[Any]) -> List[FeedEntry]:
        """Get content of articles from given entries of the news articles.
        Is meant to be used after get_entries_by_date for rss feeds which do not provide full content in their entries.

        Args:
            entries (list): List of entries to get content from

        Returns:
            list: List of entries with content
        """
        pass

    @abstractmethod
    def create_feed_from_feed_entries(self, entries: List[FeedEntry]) -> Feed:
        pass

    @abstractmethod
    def create_todays_feed(self) -> Feed:
        pass
