import argparse
import datetime
import os

import pdfkit
import pdfplumber

# python html_to_pdf.py -u 'https://***'
# python html_to_pdf.py -u 'https://***' -o output

options = {
    'page-size': 'A4',
    'encoding': 'UTF-8',
    'no-outline': None
}

DATA_DIR = './data'
TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))  # UTC+8


def add_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', '-u', type=str, required=True,
                        help='Enter the URL you want to transform to PDF.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Enter the output file name. If omitted, a timestamped name will be used.')
    return parser.parse_args()


def html_to_pdf_text(url: str, output_name: str = None):
    now = datetime.datetime.now(tz=TIMEZONE)
    filename_base = output_name or now.strftime('%Y-%m-%d_%H-%M-%S')

    os.makedirs(DATA_DIR, exist_ok=True)

    pdf_path = os.path.join(DATA_DIR, f'{filename_base}.pdf')
    txt_path = os.path.join(DATA_DIR, f'{filename_base}.txt')

    # Step 1: Convert HTML to PDF
    print("Converting webpage to PDF...")
    pdfkit.from_url(url, pdf_path, options=options)

    # Step 2: Extract text from PDF
    print("Extracting text from PDF...")
    with pdfplumber.open(pdf_path) as pdf:
        full_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

    # Step 3: Save extracted text
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(full_text)

    print(f"Done! PDF saved to: {pdf_path}")
    print(f"Extracted text saved to: {txt_path}")


if __name__ == "__main__":
    args = add_parse_args()
    html_to_pdf_text(args.url.strip(), args.output.strip()
                     if args.output else None)
