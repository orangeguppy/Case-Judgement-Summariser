import requests
from bs4 import BeautifulSoup

# Scrape all pages (930), collect and combine all urls leading to individual judgement cases from all the pages
# Link being scraped: https://www.elitigation.sg/gd (this code iterates through all pages in this link by changing the CurrentPage variable in the url)
def scrape_all_pages(num_pages):
    counter = 14000
    print("Scraping URLs")
    urls = []
    for i in range(700, 932):
        print(i)
        # ...CurrentPage={i}...: i represents the page number to scrape from
        url = f"https://www.elitigation.sg/gd/Home/Index?Filter=SUPCT&YearOfDecision=All&SortBy=DateOfDecision&CurrentPage={i}&SortAscending=False&PageSize=0&Verbose=False&SearchQueryTime=0&SearchTotalHits=0&SearchMode=True&SpanMultiplePages=False"
        scraped_urls = extract_urls_from_page(url)
        compile_judgements(scraped_urls, counter)
        counter += 11
    print("Done scraping URLs")
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
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    case_summary = soup.find("div", id="divCaseSummary")
    if case_summary.find("strong") is not None: # If there is no case summary, don't return anything
        print("Summary")
        print(url)
        judgement = soup.find("div", id="divJudgement").text
        return judgement, case_summary.text
    else:
        print("No summary")
        return None

# Scrape all judgements from a set of urls
def compile_judgements(urls, counter):
    judgements = []
    for i in range(len(urls)):
        judgement = scrape_judgement("https://www.elitigation.sg" + urls[i])
        if (judgement is not None):
            # Write to dataset
            with open(f"dataset/sg/judgement/{i+1358+counter}.txt", 'w', encoding='utf-8') as file:
                file.write(judgement[0])
            with open(f"dataset/sg/summary/{i+1358+counter}.txt", 'w', encoding='utf-8') as file:
                file.write(judgement[1])
            judgements.append(judgement)
    return judgements

# Scrape a specified number of pages (includes pagination)
num_pages_to_scrape = 931
urls = scrape_all_pages(num_pages_to_scrape)
print(urls)
judgements = compile_judgements(urls)