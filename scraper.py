import requests
from bs4 import BeautifulSoup
import os
import certifi
import pdfplumber
import re

urls = [
    "https://ao.ttu.edu.tw/",
    "https://ao.ttu.edu.tw/p/412-1073-2399.php",
    "https://www.ttu.edu.tw/",
    "https://oia.ttu.edu.tw/",
    "https://ao.ttu.edu.tw/p/412-1073-2308.php",
    "https://ao.ttu.edu.tw/p/412-1073-2310.php",
    "https://ao.ttu.edu.tw/p/412-1073-2312.php",
    "https://ao.ttu.edu.tw/p/412-1073-2311.php",
    "https://ao.ttu.edu.tw/p/412-1073-2477.php",
    "https://ao.ttu.edu.tw/p/412-1073-3168.php",
    "https://ao.ttu.edu.tw/p/405-1073-15024,c2403.php",
    "https://ao.ttu.edu.tw/p/404-1073-37644.php",
    "https://ao.ttu.edu.tw/p/405-1073-15025,c2403.php",
    "https://ao.ttu.edu.tw/p/412-1073-2405.php",
    "https://ao.ttu.edu.tw/p/412-1073-2680.php",
    "https://ao.ttu.edu.tw/p/412-1073-2637.php",
    "https://ao.ttu.edu.tw/p/412-1073-2387.php",
    "https://ao.ttu.edu.tw/p/412-1073-2388.php",
    "https://ao.ttu.edu.tw/p/412-1073-2390.php",
    "https://ao.ttu.edu.tw/var/file/73/1073/img/328/(TTU)TatungUniversityFactSheet2023.pdf",
    "https://ao.ttu.edu.tw/var/file/73/1073/img/323/Applicationformforexchangestudents(Regularsemesterterm).pdf",
    "https://ao.ttu.edu.tw/var/file/73/1073/img/326/737320368.pdf",
]

os.makedirs("data", exist_ok=True)

def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove garbled/non-ASCII heavy lines
        non_ascii = sum(1 for c in line if ord(c) > 127)
        if non_ascii > 3:
            continue
        # Remove UI junk
        junk = [
            "search", "menu", "home", "ok", "cancel", "close (esc)",
            "share", "toggle fullscreen", "zoom in/out", "more", "faq",
            "video", "quick links", "previous (arrow left)", "next (arrow right)",
            "contact us", "top news", "about ttu",
            "academics", "exchange students", "international degree students"
        ]
        if line.lower() in junk:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def fetch(url):
    try:
        response = requests.get(url, verify=certifi.where(), timeout=15)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = requests.get(url, verify=False, timeout=15)
    return response

def scrape_html(url):
    response = fetch(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav, header, footer, scripts, styles
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return clean_text(text)

def scrape_pdf(url):
    response = fetch(url)
    temp_path = "temp_pdf.pdf"
    with open(temp_path, "wb") as f:
        f.write(response.content)

    text = ""
    with pdfplumber.open(temp_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    os.remove(temp_path)
    return clean_text(text)

for i, url in enumerate(urls):
    print(f"\nProcessing ({i}): {url}")
    try:
        if url.endswith(".pdf"):
            text = scrape_pdf(url)
        else:
            text = scrape_html(url)

        if len(text.strip()) < 100:
            print(f"Skipping — too little content after cleaning.")
            continue

        file_path = os.path.join("data", f"page{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {file_path} ({len(text)} chars)")

    except Exception as e:
        print(f"Error: {e}")
        continue

print("\nDone")