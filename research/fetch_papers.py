import urllib.request
import re
import os
import time

urls = [
    "https://arxiv.org/abs/2504.02898",
    "https://arxiv.org/abs/2503.22503",
    "https://arxiv.org/abs/2505.00579",
    "https://arxiv.org/abs/2504.06753",
    "https://arxiv.org/abs/2203.16263",
    "https://arxiv.org/abs/2308.14970",
    "https://arxiv.org/abs/2507.21463",
    "https://arxiv.org/abs/2404.15143",
    "https://arxiv.org/abs/2406.06086",
    "https://arxiv.org/abs/2404.13892"
]

output_dir = "research/audio_deepfake"
summary_file = os.path.join(output_dir, "SUMMARY.md")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def download_file(url, filepath):
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_info(html):
    # Title: <h1 class="title mathjax"><span class="descriptor">Title:</span> ... </h1>
    title_match = re.search(r'<h1 class="title mathjax"><span class="descriptor">Title:</span>(.*?)</h1>', html, re.DOTALL)
    title = clean_text(title_match.group(1)) if title_match else "Unknown Title"

    # Abstract: <blockquote class="abstract mathjax"> <span class="descriptor">Abstract:</span> ... </blockquote>
    abstract_match = re.search(r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>(.*?)</blockquote>', html, re.DOTALL)
    abstract = clean_text(abstract_match.group(1)) if abstract_match else "Unknown Abstract"

    return title, abstract

with open(summary_file, "w") as f:
    f.write("# Research Summary: AI Voice Detection\n\n")

    for url in urls:
        print(f"Processing {url}...")
        html = fetch_url(url)
        if not html:
            continue

        title, abstract = extract_info(html)

        # PDF URL
        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
        pdf_filename = url.split("/")[-1] + ".pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        print(f"  Title: {title}")
        print(f"  Downloading PDF to {pdf_path}...")

        if download_file(pdf_url, pdf_path):
            print("  Download success.")
        else:
            print("  Download failed.")

        f.write(f"## [{title}]({url})\n\n")
        f.write(f"**Abstract**:\n{abstract}\n\n")
        f.write(f"**PDF**: [Local Copy]({pdf_filename})\n\n")
        f.write("---\n\n")

        # Be polite to the server
        time.sleep(1)

print(f"Done. Summary written to {summary_file}")
