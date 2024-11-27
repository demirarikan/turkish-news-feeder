from datetime import datetime
from typing import Any, List

from base_rss_parser import BaseRssParser

from models import Feed, FeedEntry


class NTVRssParser(BaseRssParser):
    def __init__(self, rss_url: str):
        """Create a new NTVRssParser object

        Args:
            rss_url (str): Url to the NTV news website rss feed
        """
        super().__init__(rss_url)

    def __sanitize_content(self, str: str) -> str:
        str = str.replace("\n", "")
        str = str.replace("\r", "")
        return str

    def create_feed_entry_from_feedparser(self, entry: Any) -> FeedEntry:
        article_url = entry.link
        html = super()._get_html(article_url)
        article_tag = html.find("article")
        if article_tag:
            h1_tag = article_tag.find("h1", class_="category-detail-title")
            h2_tag = article_tag.find("h2", class_="category-detail-sub-title")
            div_tag = article_tag.find("div", class_="category-detail-content-inner")

            if not h1_tag or not h2_tag or not div_tag:
                raise ValueError("article does not have necessary tags!")
            else:
                title = h1_tag.get_text()
                subtitle = h2_tag.get_text()
                content = subtitle + "\n" + div_tag.get_text()
                # remove \n and \r
                content = self.__sanitize_content(content)

            return FeedEntry(header=title, content=content, link=article_url)
        else:
            raise ValueError("No <article> tag found in the article!")

    def create_feed_from_feed_entries(self, feed_entries: List[FeedEntry]) -> Feed:
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
