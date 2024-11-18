import re
from datetime import datetime

import feedparser
from main import Feed, FeedEntry

class RSSParser:
    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.rss_feed = feedparser.parse(rss_url)

    def html_strip(self, html_str: str) -> str:
        return re.sub(r"<.*?>", "", html_str)

    def get_entries_by_date(self, date: datetime) -> list:
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
    
    def serialize_to_feed(self, entries: list) -> Feed:
        feed_entries = []
        for entry in entries:
            feed_entries.append(
                FeedEntry(
                    header=entry.title,
                    content=self.html_strip(entry.content[0]["value"]),
                    link=entry.link,
                )
            )
        return Feed(website=self.rss_url, entries=feed_entries)



if __name__ == "__main__":
    # will need a better way to save files
    # but a list will do for now
    rss_url = "https://www.ntv.com.tr/gundem.rss"
    parser = RSSParser(rss_url)
    entries = parser.get_entries_by_date(datetime.now())
    feed = parser.serialize_to_feed(entries)
    print(feed.website, feed.entries[0].content)
