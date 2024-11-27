from datetime import datetime
from typing import Any, List

from base_rss_parser import BaseRssParser

from models import Feed, FeedEntry


class BBCRssParser(BaseRssParser):
    def __init__(self, rss_url: str):
        """Create a new BBCRssParser object

        Args:
            rss_url (str): Url to the BBC news website rss feed
        """
        super().__init__(rss_url)

    def create_feed_entry_from_feedparser(self, entry: Any) -> FeedEntry:
        article_url = entry.link
        html = super()._get_html(article_url)
        main_tag = html.find("main")
        # print(main_tag)
        content_tags = main_tag.findChildren(
            "div", class_="bbc-19j92fr ebmt73l0", recursive=False
        )
        content = ""
        for content_tag in content_tags:
            content += content_tag.get_text()
        return FeedEntry(header=entry.title, content=content, link=article_url)

    def create_feed_from_feed_entries(self, feed_entries: List[Any]) -> Feed:
        return Feed(website=self.rss_url, entries=feed_entries)

    def create_todays_feed(self) -> Feed:
        entries = self.get_entries_by_date(datetime.now())
        feed_entries = []
        for entry in entries:
            feed_entry = self.create_feed_entry_from_feedparser(entry)
            feed_entries.append(feed_entry)
        return self.create_feed_from_feed_entries(feed_entries)
