from datetime import datetime
from typing import Any, List

from base_rss_parser import BaseRssParser

from src.main import Feed, FeedEntry


class SozcuRssParser(BaseRssParser):
    def __init__(self, rss_url: str):
        """Create a new SozcuRssParser object

        Args:
            rss_url (str): Url to the Sozcu news website rss feed
        """
        super().__init__(rss_url)

    def __sanitize_content(self, content: str) -> str:
        content = content.replace("\n", "")
        content = content.replace("\r", "")
        return content

    def create_feed_entry_from_feedparser(self, entry: Any) -> FeedEntry:
        article_url = entry.link
        html = super()._get_html(article_url)
        article_body_tag = html.find("div", class_="article-body")
        if article_body_tag:
            content = article_body_tag.get_text()
            content = self.__sanitize_content(content)
            return FeedEntry(header=entry.title, content=content, link=article_url)
        else:
            raise ValueError("No <div class:article-vody> tag found in the article!")

    def create_feed_from_feed_entries(self, feed_entries: List[Any]) -> Feed:
        return Feed(website=self.rss_url, entries=feed_entries)

    def create_todays_feed(self) -> Feed:
        entries = self.get_entries_by_date(datetime.now())
        feed_entries = []
        for entry in entries:
            try:
                feed_entry = self.create_feed_entry_from_feedparser(entry)
            except ValueError:
                continue
            feed_entries.append(feed_entry)
        feed = self.create_feed_from_feed_entries(feed_entries)
        return feed

