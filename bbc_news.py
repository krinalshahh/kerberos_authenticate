import requests
from bs4 import BeautifulSoup
from collections import deque
import time
from urllib.parse import urljoin, urlparse
from requests_kerberos import HTTPKerberosAuth
import random


class InternalNewsCrawler:
    def __init__(self, starting_url, max_articles=10, max_depth=2, delay=1):
        self.starting_url = starting_url
        self.max_articles = max_articles
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls = set()
        self.queue = deque()
        self.articles = []
        self.domain = urlparse(self.starting_url).netloc

        # Initialize session with Kerberos authentication
        self.session = requests.Session()
        self.session.auth = HTTPKerberosAuth()

        # Standard headers (no fake user agent)
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return (parsed.netloc == self.domain
                and url not in self.visited_urls
                and not any(ext in parsed.path.lower()
                            for ext in ['.pdf', '.jpg', '.png', '.jpeg', '.gif', '.mp4', '.zip', '.doc', '.xls']))

    def is_article_page(self, soup):
        article = soup.find('article') or soup.find('div', role='article')
        headline = soup.find('h1') or soup.find('title')
        date_published = (soup.find('time') or
                          soup.find('meta', property='article:published_time') or
                          soup.find('span', class_='date'))
        return bool(article and headline and date_published)

    def extract_article_content(self, soup):
        article = (soup.find('article') or
                   soup.find('div', class_='article-content') or
                   soup.find('div', class_=lambda x: x and 'article' in x.lower()) or
                   soup.find('div', class_=lambda x: x and 'content' in x.lower()) or
                   soup.find('main'))

        if not article:
            return None

        # Clean up the content
        for element in article.find_all(['script', 'style', 'nav', 'footer', 'aside',
                                         'figure', 'img', 'iframe', 'form', 'button']):
            element.decompose()

        for div in article.find_all('div', class_=lambda x: x and any(
                cls in x.lower() for cls in ['share', 'related', 'advert', 'comments',
                                             'author', 'sidebar', 'recommendations', 'newsletter'])):
            div.decompose()

        return article.get_text(separator='\n', strip=True)

    def process_page(self, url, depth):
        try:
            # Random delay to mimic human behavior
            time.sleep(self.delay * random.uniform(0.5, 1.5))

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            if response.status_code == 401:
                print("Authentication failed - possibly not logged in")
                return

            soup = BeautifulSoup(response.text, 'html.parser')
            self.visited_urls.add(url)

            if self.is_article_page(soup):
                content = self.extract_article_content(soup)
                if content:
                    title = (soup.find('h1') or soup.find('title'))
                    title = title.get_text(strip=True) if title else "No title"

                    date = (soup.find('time') or
                            soup.find('meta', property='article:published_time') or
                            soup.find('span', class_='date'))
                    date = (date.get('datetime') if date and date.get('datetime')
                            else date.get_text(strip=True) if date else "Unknown")

                    self.articles.append({
                        'url': url,
                        'title': title,
                        'date': date,
                        'content': content[:5000] + "..." if len(content) > 5000 else content,
                        'depth': depth
                    })

                    print(f"Article found: {title[:60]}...")

            if depth < self.max_depth and len(self.articles) < self.max_articles:
                links = soup.find_all('a', href=True)
                random.shuffle(links)

                for link in links:
                    absolute_url = urljoin(url, link['href'])
                    if self.is_valid_url(absolute_url):
                        self.queue.append((absolute_url, depth + 1))

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

    def crawl(self):
        print(f"\nStarting crawl on {self.starting_url}")
        print("Using Kerberos authentication - will only access pages where automatically logged in")

        # Verify initial authentication
        try:
            test_response = self.session.get(self.starting_url, timeout=10)
            if test_response.status_code == 401:
                print("ERROR: Not authenticated to access this site")
                return
        except Exception as e:
            print(f"Initial connection failed: {str(e)}")
            return

        self.queue.append((self.starting_url, 0))

        while self.queue and len(self.articles) < self.max_articles:
            url, depth = self.queue.popleft()
            self.process_page(url, depth)

        print(f"\nCrawl completed. Found {len(self.articles)} articles")

    def save_results(self, filename="crawled_articles.json"):
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")

    def print_summary(self):
        print("\nCrawl Summary:")
        print(f"Total articles found: {len(self.articles)}")
        if self.articles:
            print("\nSample articles:")
            for article in self.articles[:3]:
                print(f"\nTitle: {article['title']}")
                print(f"URL: {article['url']}")
                print(f"Date: {article['date']}")
                print(f"Preview: {article['content'][:100]}...")


if __name__ == "__main__":
    # Replace these with your actual internal sites
    internal_sites = {
        'Company Intranet News': 'https://news.internal.company.com',
        'Enterprise Portal': 'https://portal.company.net/news',
        'Department News Hub': 'https://dept.organization.org/news'
    }

    print("Available internal sites (Kerberos-authenticated):")
    for i, (name, url) in enumerate(internal_sites.items(), 1):
        print(f"{i}. {name} - {url}")

    try:
        choice = int(input("\nSelect a site to crawl (1-3): ")) - 1
        selected_url = list(internal_sites.values())[choice]

        crawler = InternalNewsCrawler(
            starting_url=selected_url,
            max_articles=15,
            max_depth=2,
            delay=2
        )

        crawler.crawl()
        crawler.print_summary()
        crawler.save_results()

    except (ValueError, IndexError):
        print("Invalid selection")
    except KeyboardInterrupt:
        print("\nCrawling stopped by user")