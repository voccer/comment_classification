import logging
import time
import re
import os
import shutil
import src.Setting as setting

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup



class SeleniumCrawler(object):

    def __init__(self):
        self.neg_file = "../Data/MyData/neg"
        self.pos_file = "../Data/MyData/pos"
        self.browser = self.get_driver()  # Add path to your Chromedriver
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

    def get_data(self, soup):
        comment_div = soup.find_all("div", class_=re.compile("lister-item-content"))

        for x in comment_div:
            self.comment.append(x.find("div", class_="content").find("div").get_text())
            if x.find("div", class_= "ipl-ratings-bar") == None:
                self.star.append(0)
            else:
                self.star.append(x.find("div", class_= "ipl-ratings-bar").find("span").find("span").get_text())

    def txt_output(self):
        count_neg = 0
        count_pos = 0
        for x, y in zip(self.star, self.comment):
            if int(x) <= 4:
                with open(self.neg_file+"/"+str(count_neg)+"_"+str(x)+".txt", mode = 'a' , encoding='utf-8') as outputfile:
                    outputfile.write(y)
                count_neg += 1
            else:
                with open(self.pos_file+"/"+str(count_pos)+"_"+str(x)+".txt", mode = 'a' , encoding='utf-8') as outputfile:
                    outputfile.write(y)
                count_pos += 1

    def txt_output_for_app(self):
        count = 0
        if os.path.isdir(setting.DIR_APP_PATH):
            shutil.rmtree(setting.DIR_APP_PATH, ignore_errors=True)
        os.makedirs(setting.DIR_APP_PATH + "/comment")
        for x, y in zip(self.star, self.comment):
            with open(setting.DIR_APP_PATH+"/comment/"+str(count)+"_"+str(x)+".txt", mode = 'a') as f:
                f.write(y)
                count += 1

    def run_crawler(self, link):
        html = self.get_page(link)
        soup = self.get_soup(html)
        if soup is not None:
            self.get_data(soup)
            # self.txt_output()
            self.txt_output_for_app()
        self.browser.close()

# Lấy dữ liệu từ nhiều link, dùng để dây dựng bộ train, test
# if __name__ == '__main__':
#     start_url = "https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt"
#     crawler = SeleniumCrawler(start_url=start_url)
#     crawler.url_queue.append("https://www.imdb.com/title/tt7401588/reviews")
#     # crawler.url_queue.append("https://www.imdb.com/title/tt1727824/reviews?ref_=tt_ov_rt")
#     try:
#         crawler.run_crawler()
#     except Exception as e:
#         crawler.browser.close()