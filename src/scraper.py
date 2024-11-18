import re
from datetime import datetime

import feedparser


class RSSParser:
    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.rss_feed = feedparser.parse(rss_url)

    def html_strip(self, html_str: str) -> str:
        return re.sub(r"<.*?>", "", html_str)

    def get_entries_by_date(self, date: datetime, strip_html=True) -> list:
        entries = []
        for entry in self.rss_feed.entries:
            # published_parsed return 9 tuple in UTC
            # we get all entries that are published on the same day
            if (
                entry.published_parsed[0] == date.year()
                and entry.published_parsed[1] == date.month()
                and entry.published_parsed[2] == date.day()
            ):
                entries.append(entry)
        return entries
    
    def serialize_to_feed(self, entries: list) -> Feed:
        



# will need a better way to save files
# but a list will do for now
rss_url = "https://www.ntv.com.tr/gundem.rss"
feed = feedparser.parse(rss_url)
print(feed.entries[0].published_parsed[0])
