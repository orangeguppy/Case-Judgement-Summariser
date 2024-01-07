import requests
from bs4 import BeautifulSoup

# Scrape all pages
def scrape_all_pages(num_pages):
    urls = []
    for i in range(1, num_pages + 1):
        url = "https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=DateOfDecision&CurrentPage={i}&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=True&SpanMultiplePages=False"
        scraped_urls = extract_urls_from_page(url)
        urls.extend(scraped_urls)
    return urls

# Get all links to judgement cases on a page
def extract_urls_from_page(url):
    # Store results
    urls = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all("a", class_="h5 gd-heardertext")
    href_links = [link.get('href') for link in results]
    return href_links

# Scrape the judgement (text form) from the page
def scrape_judgement(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    judgement = soup.find("div", id="divJudgement").text
    return judgement

# Scrape all judgements
def compile_judgements(urls):
    judgements = []
    for url in urls:
        judgement = scrape_judgement("https://www.elitigation.sg" + url)
        judgements.append(judgement)
    return judgements

# Scrape a specified number of pages (includes pagination)
num_pages_to_scrape = 2
urls = scrape_all_pages(num_pages_to_scrape)
judgements = compile_judgements(urls)
print(judgements[2])