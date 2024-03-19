from flask import Flask, jsonify

app = Flask(__name__)
# pip install requests
# pip install beautifulsoup4

#Issue HTTPS request to page, and retrieves data in a Python object
import requests
URL = "https://www.judiciary.gov.sg/news-and-resources/glossary"
page = requests.get(URL)

# Inputs page HTML and parse the code for webscraping
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, "html.parser")

#Extracts code with id="Body_TF0229BA3008_Col00"
body = soup.find(id="Body_TF0229BA3008_Col00")

#Extracts all HTML with class="headless-table"
tables = body.find_all("div", class_="headless-table")

#Extracts all HTML with tr
rows = soup.find_all("tr")

#Extracts the data, stores in dictionary
glossary_dict = {}
for row in rows:
    td = row.find_all("td")
    glossary_dict[td[0].get_text().lower()] = td[1].get_text(strip=True)
print(glossary_dict)

@app.route("/api/glossary")
def get_glossary():
    return jsonify(glossary_dict)

if __name__ == "__main__":
    app.run(debug=True)

