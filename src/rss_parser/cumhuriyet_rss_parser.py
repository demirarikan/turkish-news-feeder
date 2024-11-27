from datetime import datetime
from typing import Any, List
from urllib.parse import urlparse

from base_rss_parser import BaseRssParser

from src.main import Feed, FeedEntry


class CumhuriyetRssParser(BaseRssParser):
    def __init__(self, rss_url):
        super().__init__(rss_url)
        self.allowed_topics = ["turkiye", "gundem", "ekonomi", "siyaset"]

    def __is_url_topic_allowed(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path
        topic = path.split("/")[1]
        return topic in self.allowed_topics

    def __sanitize_content(self, content: str) -> str:
        content = content.replace("\n", "")
        content = content.replace("\r", "")
        content = content.replace("\xa0", "")
        return content

    def create_feed_entry_from_feedparser(self, entry: Any) -> FeedEntry:
        article_url = entry.link
        if not self.__is_url_topic_allowed(article_url):
            raise ValueError("Article topic is not allowed!")
        html = super()._get_html(article_url)
        article_subheader = html.find("h2", class_="spot").get_text()
        article_content = html.find("div", class_="haberMetni").get_text()
        article_content = article_subheader + " " + article_content
        article_content = self.__sanitize_content(article_content)
        return FeedEntry(header=entry.title, content=article_content, link=article_url)

    def create_feed_from_feed_entries(self, entries: List[FeedEntry]) -> Feed:
        return Feed(website=self.rss_url, entries=entries)

    def create_todays_feed(self) -> Feed:
        entries = self.get_entries_by_date(datetime.now())
        feed_entries = []
        for entry in entries:
            try:
                feed_entry = self.create_feed_entry_from_feedparser(entry)
            except ValueError:
                continue
            feed_entries.append(feed_entry)
        return self.create_feed_from_feed_entries(feed_entries)
