# src/ingest_domains.py

import os
import glob
import argparse
from pypdfium2 import PdfDocument
import pytesseract
import pandas as pd
from PIL import Image
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metadata(text: str, pdf_path: str) -> dict:
    """
    Extract basic metadata from PDF text and filename.
    Returns a dict with title, date, and any detected document type.
    """
    filename = os.path.basename(pdf_path)
    metadata = {
        "filename": filename,
        "title": "",
        "created_at": "",
        "doc_type": ""
    }
    
    # Try to extract title - usually in first few lines
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines[:10] if line.strip()]
    if non_empty_lines:
        metadata["title"] = non_empty_lines[0]
    
    # Extract date patterns
    date_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            metadata["created_at"] = dates[0]
            break
    
    # Try to detect document type
    doc_types = {
        "regulation": ["regulation", "directive", "law", "statute", "ordinance", "code", "rule", "legal"],
        "report": ["report", "assessment", "analysis", "study", "survey", "review", "evaluation"]
    }
    
    for doc_type, keywords in doc_types.items():
        lower_text = text.lower()
        for keyword in keywords:
            if keyword in lower_text:
                metadata["doc_type"] = doc_type
                break
        if metadata["doc_type"]:
            break
    
    return metadata

def enhance_image_for_ocr(image):
    """
    Apply image preprocessing to improve OCR results.
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Increase contrast and apply simple thresholding
    # This is a basic approach - could be enhanced with more sophisticated techniques
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast
    
    return image

def pdf_to_text(path: str) -> str:
    """
    Extracts text from a PDF file. First attempts embedded text extraction;
    if that yields no text, falls back to Tesseract OCR on each page image.
    Includes image enhancement for better OCR results.
    """
    doc = None
    try:
        doc = PdfDocument(path)
        texts = []
        logger.info(f"Processing PDF: {path}")
        try:
            for page_idx, page in enumerate(doc):
                try:
                    # Try embedded text first
                    raw = page.get_textpage().get_text_range()
                    if raw and len(raw.strip()) > 50:  # Check if reasonable amount of text
                        logger.debug(f"Page {page_idx+1}: Using embedded text")
                        texts.append(raw)
                    else:
                        # Fallback to OCR
                        logger.debug(f"Page {page_idx+1}: Using OCR")
                        try:
                            bitmap = page.render(
                                scale=2.0,  # Increase resolution for better OCR
                                rotation=0,
                                crop=None
                            )
                            pil_img = bitmap.to_pil()
                            # Enhance image for better OCR results
                            enhanced_img = enhance_image_for_ocr(pil_img)
                            ocr_text = pytesseract.image_to_string(
                                enhanced_img, 
                                config='--oem 3 --psm 6'
                            )
                            texts.append(ocr_text)
                        except Exception as e:
                            logger.error(f"OCR error on page {page_idx+1}: {str(e)}")
                            texts.append(f"[OCR ERROR ON PAGE {page_idx+1}]")
                except Exception as e:
                    logger.error(f"Error processing page {page_idx+1}: {str(e)}")
                    texts.append(f"[ERROR PROCESSING PAGE {page_idx+1}]")
        except Exception as e:
            logger.error(f"Error iterating through PDF pages: {str(e)}")
            try:
                logger.info(f"Trying to recover. PDF has {len(doc)} pages.")
                texts.append(f"[PDF PROCESSING ERROR: {str(e)}]")
            except:
                texts.append(f"[SEVERE PDF ERROR: {str(e)}]")
        full_text = "\n".join(texts) if texts else f"[FAILED TO EXTRACT TEXT FROM {path}]"
        logger.info(f"Extracted {len(full_text)} characters from {path}")
        return full_text
    except Exception as e:
        logger.error(f"Critical error processing PDF {path}: {str(e)}")
        return f"[CRITICAL PDF ERROR: {str(e)}]"
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception as e:
                logger.warning(f"Error closing PDF document: {str(e)}")


def ingest_domain(domain: str, input_dir: str, output_dir: str):
    """
    Reads all PDFs in a domain subfolder, converts to text, and writes a Parquet
    file named <domain>.parquet in the output directory.

    Args:
        domain: Name of the domain (e.g. "contracts").
        input_dir: Root folder containing domain subdirectories.
        output_dir: Directory where Parquet shards will be written.
    """
    pdf_paths = glob.glob(os.path.join(input_dir, domain, "*.pdf"))
    records = []
    
    logger.info(f"Found {len(pdf_paths)} PDFs in {domain} directory")

    for pdf_idx, pdf in enumerate(pdf_paths):
        logger.info(f"Processing {pdf_idx+1}/{len(pdf_paths)}: {pdf}")
        try:
            # Extract text content
            text = pdf_to_text(pdf)
            
            # Extract metadata
            metadata = extract_metadata(text, pdf)
            
            records.append({
                "file_path": pdf,
                "filename": os.path.basename(pdf),
                "domain": domain,
                "text": text,
                "title": metadata["title"],
                "created_at": metadata["created_at"],
                "detected_type": metadata["doc_type"]
            })
            
        except Exception as e:
            logger.error(f"Error processing {pdf}: {str(e)}")

    if records:
        df = pd.DataFrame(records)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{domain}.parquet")
        df.to_parquet(output_file)
        logger.info(f"Wrote {len(records)} documents to {output_file}")
    else:
        logger.warning(f"No documents successfully processed for {domain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents from multiple domains into Parquet shards"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Root directory containing domain-named subfolders with PDFs"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for Parquet shards"
    )
    parser.add_argument(
        "--domains", nargs="+", required=True,
        help="List of domain names to ingest, e.g. contracts reports regulations"
    )
    args = parser.parse_args()

    for domain in args.domains:
        ingest_domain(domain, args.input_dir, args.output_dir)
