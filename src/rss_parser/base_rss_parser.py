import re
from abc import ABC, abstractmethod
from datetime import datetime
from src.main import Feed, FeedEntry
from typing import List, Any

class BaseRssParser(ABC):
    def __init__(self, rss_url: str):
        self.rss_url = rss_url

    def html_strip(self, html_str: str) -> str:
        return re.sub(r"<.*?>", "", html_str)
    
    @abstractmethod
    def get_entries_by_date(self, date: datetime) -> List[Any]:
        """Get entries that are on the same day as the given date

        Args:
            date (datetime): Date to collect entries

        Returns:
            list: List of feedperser entries that are on the same day as the given date
        """
        pass

    @abstractmethod
    def create_feed_entry_from_feedparser(self, entry: List[Any]) -> List[Any]:
        """Get content of articles from given entries of the news articles.
        Is meant to be used after get_entries_by_date for rss feeds which do not provide full content in their entries.

        Args:
            entries (list): List of entries to get content from

        Returns:
            list: List of entries with content
        """
        pass

    @abstractmethod
    def create_feed_from_feed_entries(self, entries: List[Any]) -> Feed:
        pass

    

