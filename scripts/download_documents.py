import os
import requests
from pathlib import Path
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDownloader:
    def __init__(self, base_dir: str = "raw"):
        self.base_dir = Path(base_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def download_file(self, url: str, save_path: Path) -> bool:
        """Download a file from URL with proper error handling and rate limiting."""
        try:
            # Add delay to respect rate limits
            time.sleep(1)
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"URL {url} might not be a PDF (content-type: {content_type})")
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded: {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    def download_documents(self, documents: Dict[str, List[str]]):
        """Download documents for each category."""
        for category, urls in documents.items():
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            for i, url in enumerate(urls, 1):
                filename = f"{category}_{i}.pdf"
                save_path = category_dir / filename
                if not save_path.exists():
                    self.download_file(url, save_path)
                else:
                    logger.info(f"File already exists: {save_path}")

def main():
    documents = {
        "contracts": [
            # UN Procurement Division Model Contract for Services
            "https://www.un.org/Depts/ptd/sites/www.un.org.Depts.ptd/files/pdf/model_contract_services.pdf",
            # EU Framework Contract for Services
            "https://ec.europa.eu/info/sites/default/files/standard_contract_services_en.pdf"
        ],
        "reports": [
            # WHO World Health Statistics 2023
            "https://cdn.who.int/media/docs/default-source/gho-documents/world-health-statistic-reports/2023/whs2023.pdf",
            # UNICEF Annual Report 2022
            "https://www.unicef.org/media/123181/file/UNICEF-Annual-Report-2022.pdf",
            # UN World Economic Situation and Prospects 2023
            "https://www.un.org/development/desa/dpad/wp-content/uploads/sites/45/WESP2023_web.pdf"
        ],
        "regulations": [
            # EU GDPR (official)
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
            # EU Medical Device Regulation
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0745",
            # UN Convention Against Corruption
            "https://www.unodc.org/documents/brussels/UN_Convention_Against_Corruption.pdf"
        ]
    }
    downloader = DocumentDownloader()
    downloader.download_documents(documents)

if __name__ == "__main__":
    main() 