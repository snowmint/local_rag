import argparse
import datetime
import os

import requests
from bs4 import BeautifulSoup

DATA_DIR = './data'
TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))  # UTC+8


def add_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', '-u', type=str, required=True,
                        help='Enter the URL you want to transform to PDF.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Enter the output file name. If omitted, a timestamped name will be used.')
    return parser.parse_args()


def get_text_from_url(url: str, output_name: str = None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')

    for script_or_style in soup(['script', 'style', 'noscript']):
        script_or_style.decompose()

    text = soup.get_text(separator='\n')
    lines = [line.strip()
             for line in text.splitlines()]  # if len(line.strip()) > 30
    full_text = '\n'.join(lines)

    now = datetime.datetime.now(tz=TIMEZONE)
    filename_base = output_name or now.strftime('%Y-%m-%d_%H-%M-%S')

    os.makedirs(DATA_DIR, exist_ok=True)
    txt_path = os.path.join(DATA_DIR, f'{filename_base}.txt')

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(full_text)

    return


if __name__ == '__main__':

    args = add_parse_args()
    get_text_from_url(args.url.strip(), args.output.strip()
                      if args.output else None)
    # html_text = get_text_from_url(url)

    print("Saved cleaned text from URL.")
