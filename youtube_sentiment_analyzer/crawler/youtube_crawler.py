import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def collect_youtube_comments(keyword, max_videos=3):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(options=options)
    comments = []

    try:
        driver.get("https://www.youtube.com/")
        time.sleep(2)
        search_box = driver.find_element(By.NAME, "search_query")
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        video_links = []
        seen = set()
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)
            video_elems = driver.find_elements(By.ID, 'video-title')
            for elem in video_elems:
                link = elem.get_attribute('href')
                if link and 'watch' in link and link not in seen:
                    video_links.append(link)
                    seen.add(link)
                if len(video_links) >= max_videos:
                    break
            if len(video_links) >= max_videos:
                break

        for url in video_links:
            driver.get(url)
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 800);")
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(1.5)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            comment_elems = soup.select('ytd-comment-thread-renderer #content-text')
            for c in comment_elems:
                text = c.text.strip()
                if text:
                    comments.append(text)
    finally:
        driver.quit()

    return comments
