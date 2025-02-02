import requests
from bs4 import BeautifulSoup
import os

# Base URL of the page where the initial links are located
base_url = 'https://fm.dk/udgivelser/'

# Function to ensure the URL is absolute
def make_absolute_url(base, link):
    if not link.startswith('http'):
        return f'{base}{link}'
    return link

# Directory to save the downloaded PDF files
download_folder = 'downloaded_pdfs'
os.makedirs(download_folder, exist_ok=True)

# Initialize page number
page_number = 1

while True:
    # Construct URL with page number
    page_url = f"{base_url}?pageNumber={page_number}"

    # Send a GET request to the current page URL
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f'No more pages to process or failed to fetch page {page_number}.')
        break

    # Parse the HTML content of the current page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all initial <a> tags with the specific class
    a_tags = soup.find_all('a', class_='results-item results-item--link results-item--has-image')

    # Process each <a> tag found
    for a_tag in a_tags:
        follow_url = make_absolute_url(base_url, a_tag['href'])

        # Send a GET request to the followed URL
        follow_response = requests.get(follow_url)
        follow_response.raise_for_status()

        # Parse the HTML content of the followed page
        follow_soup = BeautifulSoup(follow_response.text, 'html.parser')

        # Find the <h2> tag and the following <ul> element
        h2 = follow_soup.find('h2', string="Hent publikationen")
        if h2:
            ul = h2.find_next('ul', class_='link-list__list p')
            if ul:
                # Extract PDF links from the <a> tags within <li> elements
                pdf_links = [make_absolute_url("https://fm.dk", a['href']) for a in ul.find_all('a', href=True) if a['href'].endswith('.pdf')]

                # Download each PDF
                for link in pdf_links:
                    try:
                        # Fetch the PDF file
                        pdf_response = requests.get(link)
                        pdf_response.raise_for_status()

                        # Save the PDF file
                        pdf_name = link.split('/')[-1]
                        with open(os.path.join(download_folder, pdf_name), 'wb') as f:
                            f.write(pdf_response.content)
                        print(f'Downloaded: {pdf_name}')
                    except requests.exceptions.RequestException as e:
                        print(f'Failed to download {link}: {e}')
            else:
                print(f'No <ul> found following the <h2> for URL: {follow_url}')
        else:
            print(f'No <h2> "Hent publikationen" found for URL: {follow_url}')

    # Check if there is a next page
    next_page_link = soup.find('a', href=lambda href: href and f"?pageNumber={page_number + 1}" in href)
    if not next_page_link:
        print('No more pages found.')
        break
    else:
        page_number += 1

print('All PDFs have been downloaded or attempted.')
