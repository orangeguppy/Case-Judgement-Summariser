import requests
from bs4 import BeautifulSoup

# Scrape all pages (930), collect and combine all urls leading to individual judgement cases from all the pages
# Link being scraped: https://www.elitigation.sg/gd (this code iterates through all pages in this link by changing the CurrentPage variable in the url)
def scrape_all_pages(num_pages):
    urls = []
    for i in range(1, num_pages + 1):
        # ...CurrentPage={i}...: i represents the page number to scrape from
        url = "https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=DateOfDecision&CurrentPage={i}&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=True&SpanMultiplePages=False"
        scraped_urls = extract_urls_from_page(url)
        urls.extend(scraped_urls)
    return urls

# Get all links to judgement cases on a single page, this page contains a list of links linking to individual judgement cases
# Example page being scraped: https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=DateOfDecision&CurrentPage=2&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=True&SpanMultiplePages=False
def extract_urls_from_page(url):
    # Store results
    urls = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all("a", class_="h5 gd-heardertext")
    href_links = [link.get('href') for link in results]
    return href_links

# Scrape the judgement (text form) from a single page (containing one judgement only)
# Example page being scraped: https://www.elitigation.sg/gd/s/2024_SGHCA_1
def scrape_judgement(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    judgement = soup.find("div", id="divJudgement").text
    return judgement

# Scrape all judgements from a set of urls
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