import re
from datetime import datetime

import feedparser
from src.main import Feed, FeedEntry
from bs4 import BeautifulSoup
import requests
from base_rss_parser import BaseRssParser
from typing import List, Any

class NTVRssParser(BaseRssParser):
    def __init__(self, rss_url: str):
        """Create a new NTVRssParser object

        Args:
            rss_url (str): Url to the NTV news website rss feed
        """
        self.rss_url = rss_url
        self.rss_feed = feedparser.parse(rss_url)

    def html_strip(self, html_str: str) -> str:
        return re.sub(r"<.*?>", "", html_str)
    
    def remove_escape_chars(self, str: str) -> str:
        str = str.replace("\n", "")
        str = str.replace("\r", "")
        return str

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
    
    def create_feed_entry_from_feedparser(self, entry: Any) -> FeedEntry:
        article_url = entry.link
        response = requests.get(article_url)
        soup = BeautifulSoup(response.text, "html.parser")
        article_tag = soup.find("article")
        if article_tag:
            h1_tag = article_tag.find('h1', class_='category-detail-title')
            h2_tag = article_tag.find('h2', class_='category-detail-sub-title')
            div_tag = article_tag.find('div', class_='category-detail-content-inner')

            if not h1_tag or not h2_tag or not div_tag:
                raise ValueError("article does not have necessary tags!")
            else:
                title = h1_tag.get_text()
                subtitle = h2_tag.get_text()
                content = subtitle + "\n" + div_tag.get_text()
                # remove \n and \r
                content = self.remove_escape_chars(content)
            
            return FeedEntry(header=title, content=content, link=article_url)
        else:
            raise ValueError("No <article> tag found in the article!")

    def create_feed_from_feed_entries(self, feed_entries: List[FeedEntry]) -> Feed:
        return Feed(website=self.rss_url, entries=feed_entries)
    
    def create_todays_feed(self):
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
        

# if __name__ == "__main__":
#     import json
#     # will need a better way to save files
#     # but a list will do for now
#     rss_url = "https://www.ntv.com.tr/gundem.rss"
#     parser = NTVRssParser(rss_url)
#     feed = parser.create_todays_feed()
#     with open("feed.json", "w", encoding="utf-8") as json_file:
#         json.dump(feed.model_dump(), json_file, indent=4, ensure_ascii=False)



