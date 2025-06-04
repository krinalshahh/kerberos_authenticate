from playwright.sync_api import sync_playwright
from collections import deque
import time
from fake_useragent import UserAgent
from urllib.parse import urljoin, urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json


class FinBertAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    def analyze_sentiment(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive = predictions[:, 0].item()
            negative = predictions[:, 1].item()
            neutral = predictions[:, 2].item()

            sentiment_scores = {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'prediction': self.get_prediction(positive, negative, neutral)
            }
            return sentiment_scores
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return None

    def get_prediction(self, positive, negative, neutral):
        if positive > negative and positive > neutral:
            return "Positive"
        elif negative > positive and negative > neutral:
            return "Negative"
        else:
            return "Neutral"


class NewsCrawler:
    def __init__(self, starting_url, no_of_articles, depth, delay):
        self.starting_url = starting_url
        self.no_of_articles = no_of_articles
        self.depth = depth
        self.delay = delay
        self.urls_visited = set()
        self.queue = deque()
        self.ua = UserAgent()
        self.articles = []
        self.domain = urlparse(self.starting_url).netloc
        self.sentiment_analyzer = FinBertAnalyzer()
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            user_agent=self.ua.random,
            viewport={'width': 1280, 'height': 1024}
        )

    def __del__(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()

    def is_url_valid(self, url):
        parsed_url = urlparse(url)
        return (parsed_url.netloc == self.domain
                and url not in self.urls_visited
                and not any(extension in url.lower()
                            for extension in ['.pdf', '.jpg', '.png', 'jpeg', '.gif', '.mp4']))

    def is_page_an_article(self, page):
        # More flexible article detection for modern news sites
        article = (page.query_selector('article') or
                   page.query_selector('[data-testid*="article"]') or
                   page.query_selector('[class*="article-body"]') or
                   page.query_selector('[class*="story-content"]'))

        headline = (page.query_selector('h1') or
                    page.query_selector('[data-testid*="headline"]') or
                    page.query_selector('[class*="article-headline"]'))

        date = (page.query_selector('time') or
                page.query_selector('[datetime]') or
                page.query_selector('[class*="date"]') or
                page.query_selector('[class*="timestamp"]'))

        return article is not None and headline is not None

    def extract_found_article(self, page):
        # Try multiple patterns for article content
        article = None
        selectors = [
            'article',
            '[data-testid*="article"]',
            '[class*="article-body"]',
            '[class*="story-content"]',
            '[class*="article-content"]',
            '[class*="post-content"]'
        ]

        for selector in selectors:
            article = page.query_selector(selector)
            if article:
                break

        if not article:
            return None

        # Clean up the article content
        for selector in ['script', 'style', 'nav', 'footer', 'aside', 'figure', 'img', 'iframe', 'form']:
            for element in article.query_selector_all(selector):
                element.evaluate('element => element.remove()')

        # Remove unwanted sections
        unwanted_selectors = [
            '[class*="related"]',
            '[class*="comments"]',
            '[class*="ad"]',
            '[class*="newsletter"]',
            '[class*="author"]',
            '[class*="social"]'
        ]

        for selector in unwanted_selectors:
            for element in article.query_selector_all(selector):
                element.evaluate('element => element.remove()')

        return article.inner_text().strip()

    def analyze_article_sentiment(self, content):
        max_chunk_length = 1000
        chunks = [content[i:i + max_chunk_length] for i in range(0, len(content), max_chunk_length)]

        sentiments = []
        for chunk in chunks:
            sentiment = self.sentiment_analyzer.analyze_sentiment(chunk)
            if sentiment:
                sentiments.append(sentiment)

        if not sentiments:
            return None

        avg_sentiment = {
            'positive': np.mean([s['positive'] for s in sentiments]),
            'negative': np.mean([s['negative'] for s in sentiments]),
            'neutral': np.mean([s['neutral'] for s in sentiments]),
            'prediction': max(set([s['prediction'] for s in sentiments]),
                            key=[s['prediction'] for s in sentiments].count)
        }
        return avg_sentiment

    def process_page(self, url, depth):
        try:
            time.sleep(self.delay)
            page = self.context.new_page()

            # Set timeout and wait for content to load
            page.goto(url, timeout=60000)
            page.wait_for_selector('body', timeout=10000)

            # Wait for either article content or a reasonable timeout
            try:
                page.wait_for_selector('article, [data-testid*="article"], [class*="article"]', timeout=5000)
            except:
                pass  # Continue even if we don't find article immediately

            print(f"Loaded page: {url}")

            if self.is_page_an_article(page):
                content = self.extract_found_article(page)
                if content:
                    title = (page.query_selector('h1') or
                             page.query_selector('[data-testid*="headline"]') or
                             page.query_selector('[class*="article-headline"]') or
                             page.query_selector('title'))
                    title = title.inner_text().strip() if title else "No title"

                    date = (page.query_selector('time') or
                            page.query_selector('[datetime]') or
                            page.query_selector('[class*="date"]') or
                            page.query_selector('[class*="timestamp"]'))
                    date = (date.get_attribute('datetime') if date and date.get_attribute('datetime')
                            else date.inner_text().strip() if date else "Unknown")

                    sentiment = self.analyze_article_sentiment(content)

                    self.articles.append({
                        'url': url,
                        'title': title,
                        'date': date,
                        'content': content[:5000] + "..." if len(content) > 5000 else content,
                        'depth': depth,
                        'sentiment': sentiment
                    })
                    print(f"Article found at depth {depth}: {title[:50]}...")
                    if sentiment:
                        print(f"Sentiment: {sentiment['prediction']}")

            if depth < self.depth and len(self.articles) < self.no_of_articles:
                # More specific link selection for news sites
                links = page.query_selector_all('a[href*="/article/"], a[href*="/news/"], a[href*="/story/"]')
                for link in links:
                    href = link.get_attribute('href')
                    if href:
                        absolute_url = urljoin(self.starting_url, href)
                        if self.is_url_valid(absolute_url):
                            self.queue.append((absolute_url, depth + 1))

            page.close()

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            if 'page' in locals():
                page.close()

    def crawl(self):
        print(f"Crawler started crawling on {self.starting_url} (max depth {self.depth})")
        self.queue.append((self.starting_url, 0))
        while self.queue and len(self.articles) < self.no_of_articles:
            url, depth = self.queue.popleft()
            if url not in self.urls_visited:
                self.process_page(url, depth)
        print(f"Crawling found {len(self.articles)} articles")

    def save_Results(self, filename="NewsArticles.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        print(f"Results saved in {filename}")

    def print_summary(self):
        print("\nCollected article summary:")
        for idx, article in enumerate(self.articles, 1):
            print(f"\n{idx}. {article['title']}")
            print(f"Depth: {article['depth']} | Date: {article['date']}")
            print(f"URL: {article['url']}")
            if article.get('sentiment'):
                sentiment = article['sentiment']
                print(f"Sentiment: {sentiment['prediction']}")
                print(f"Positive: {sentiment['positive']:.2f}, Negative: {sentiment['negative']:.2f}, Neutral: {sentiment['neutral']:.2f}")
            print(f"Preview: {article['content'][:100]}...")


if __name__ == "__main__":
    sites_list = {
        'Reuters News': 'https://www.nytimes.com/',
        'Hong Kong Business': 'https://hongkongbusiness.hk',
        'SCMP News': 'https://www.scmp.com/',
        'BBC News': 'https://www.bbc.com/news'
    }

    for idx, (name, url) in enumerate(sites_list.items(), 1):
        print(f"{idx}: {name} ({url})")

    choice = int(input("\nSelect a site to crawl (1-4): ")) - 1
    selected_url = list(sites_list.values())[choice]

    crawler = NewsCrawler(
        starting_url=selected_url,
        no_of_articles=10,
        depth=2,
        delay=1
    )

    crawler.crawl()
    crawler.print_summary()
    crawler.save_Results()