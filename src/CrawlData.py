import logging
import csv
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from urllib.parse import urldefrag, urljoin
from collections import deque
from bs4 import BeautifulSoup
import os
import re

class SeleniumCrawler(object):

    def __init__(self, base_url = None, exclusion_list = None, output_file=None, start_url=None):
        # assert isinstance(exclusion_list, list), 'Exclusion list - needs to be a list'

        self.txt_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "example.txt")

        self.neg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "neg")

        self.pos_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pos")

        self.output_source = os.path.join(os.path.dirname(os.path.realpath(__file__)), "source.txt")

        self.browser = self.get_driver()  # Add path to your Chromedriver

        self.base = "https://www.imdb.com"

        self.start = start_url if start_url else base_url  # If no start URL is passed use the base_url

        self.exclusions = exclusion_list  # List of URL patterns we want to exclude

        self.crawled_urls = []  # List to keep track of URLs we have already visited

        self.url_queue = deque([self.start])  # Add the start URL to our list of URLs to crawl

        self.star = []

        self.comment = []

    def get_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=%s" % "1920,1080")
        browser = webdriver.Chrome(chrome_options=chrome_options)
        return browser

    def get_page(self, url):
        try:
            self.browser.get(url)
            wait = WebDriverWait(self.browser, 10)
            while True:
                e1 = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(),'Load More')]")))
                if not e1.is_displayed():
                    count = 0
                    while (not e1.is_displayed()) and count < 10:
                        time.sleep(.2)
                        count += 1
                        print(count)
                if e1.is_displayed():
                    e1.click()
                else: break

            return self.browser.page_source
        except Exception as e:
            logging.exception(e)
            return

    def get_soup(self, html):
        if html is not None:
            soup = BeautifulSoup(html, 'lxml')
            return soup
        else:
            return

    def get_links(self, soup):
        for link in soup.find_all('a', href=True):  # All links which have a href element
            link = link['href']  # The actually href element of the link
            if any(e in link for e in self.exclusions):  # Check if the link matches our exclusion list
                continue  # If it does we do not proceed with the link
            url = urljoin(self.base, urldefrag(link)[0])  # Resolve relative links using base and urldefrag
            if url not in self.url_queue and url not in self.crawled_urls:  # Check if link is in queue or already crawled
                if url.startswith(self.base):  # If the URL belongs to the same domain
                    self.url_queue.append(url)  # Add the URL to our queue

    def get_data(self, soup):
        comment_div = soup.find_all("div", class_=re.compile("lister-item-content"))

        for x in comment_div:
            self.comment.append(x.find("div", class_="content").find("div").get_text())
            if x.find("div", class_= "ipl-ratings-bar") == None:
                self.star.append(0)
            else:
                self.star.append(x.find("div", class_= "ipl-ratings-bar").find("span").find("span").get_text())

    def csv_output(self, url, comment):
        with open(self.csv_file, mode ='a', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile)
            for cmt in comment:
                writer.writerow([cmt])

    def txt_output(self):
        count_neg = 0
        count_pos = 0
        for x, y in zip(self.star, self.comment):
            if int(x) <= 6:
                with open(self.neg_file+"/"+str(count_neg)+"_"+str(x)+".txt", mode = 'a' , encoding='utf-8') as outputfile:
                    outputfile.write(y)
                count_neg += 1
            else:
                try:
                    with open(self.pos_file+"/"+str(count_pos)+"_"+str(x)+".txt", mode = 'a' , encoding='utf-8') as outputfile:
                        outputfile.write(y)
                    count_pos += 1
                except Exception as e:
                    print(y)
                    print(e)


    def source_output(self, source):
        with open(self.output_source, mode = 'a', encoding='utf-8') as outputsource:
            outputsource.write(source)

    def run_crawler(self):
        while len(self.url_queue):  # If we have URLs to crawl - we crawl
            current_url = self.url_queue.popleft()  # We grab a URL from the left of the list
            self.crawled_urls.append(current_url)  # We then add this URL to our crawled list
            html = self.get_page(current_url)
            if self.browser.current_url != current_url:  # If the end URL is different from requested URL - add URL to crawled list
                self.crawled_urls.append(current_url)
            soup = self.get_soup(html)
            if soup is not None:  # If we have soup - parse and write to our csv file
                # self.get_links(soup)
                self.get_data(soup)
                self.txt_output()
        self.browser.close()

if __name__ == '__main__':
    start_url = "https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt"
    crawler = SeleniumCrawler(start_url=start_url)
    crawler.url_queue.append("https://www.imdb.com/title/tt7401588/reviews")
    crawler.url_queue.append("https://www.imdb.com/title/tt1727824/reviews?ref_=tt_ov_rt")
    try:
        crawler.run_crawler()
    except Exception as e:
        crawler.browser.close()