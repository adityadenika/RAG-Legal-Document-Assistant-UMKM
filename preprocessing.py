import os
import json
import pandas as pd
import PyPDF2
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LegalElement:
    """Data class untuk elemen legal (Pasal, ayat, poin)"""
    pasal: Optional[int] = None
    ayat: Optional[int] = None
    poin: Optional[str] = None
    content: str = ""
    element_type: str = ""


class ImprovedLegalDocumentPreprocessor:
    """Preprocessor untuk dokumen legal UMKM"""

    def __init__(self):
        self.setup_patterns()
        self.debug_mode = True  # untuk debugging

    def setup_patterns(self):
        """Setup regex patterns fleksibel"""

        self.removal_patterns = {
            'header_footer': [
                r'PRES\s*IDEN[\s\n]*REPUBLIK[\s\n]*INDONESIA',
                r'SK\s*No\s*\d+\s*[A-Z]*',
                r'SALINAN',
                r'ttd\.',
                r'Salinan sesuai dengan aslinya',
                r'KEMENTERIAN.*',
                r'Djaman',
                r'YASONNA.*LAOLY',
                r'JOKO\s*WIDODO',
                r'-\s*\d+\s*-',
                r'PRESIDEN REPUBLIK INDONESIA',
                r'MENTERI.*REPUBLIK INDONESIA',
            ],
            'document_metadata': [
                r'LEMBARAN NEGARA.*NOMOR.*',
                r'TAMBAHAN LEMBARAN NEGARA.*',
                r'Diundangkan di Jakarta.*',
                r'Ditetapkan di Jakarta.*',
                r'pada tanggal.*\d{4}',
            ],
            'irrelevant_sections': [
                r'PENJELASAN\s*ATAS.*',
                r'Cukup jelas\.',
                r'Yang dimaksud dengan.*adalah.*',
                r'Agar setiap orang mengetahuinya.*',
            ]
        }

        self.structure_patterns = {
            # Pasal: mendukung berbagai format (case insensitive, dengan/tanpa titik)
            'pasal': [
                r'(?:^|\n)\s*PASAL\s+(\d+)\s*\.?\s*(?:\n|$)',
                r'(?:^|\n)\s*Pasal\s+(\d+)\s*\.?\s*(?:\n|$)',
                r'(?:^|\n)\s*pasal\s+(\d+)\s*\.?\s*(?:\n|$)',
            ],

            # Ayat: berbagai format penomoran
            'ayat': [
                r'^\s*\((\d+)\)\s*',  # (1), (2), etc.
                r'^\s*ayat\s*\((\d+)\)\s*',  # ayat (1)
                r'^\s*Ayat\s*\((\d+)\)\s*',  # Ayat (1)
            ],

            # Poin: huruf dengan berbagai format
            'poin': [
                r'^\s*([a-z])\.\s*',  # a., b., c.
                r'^\s*([a-z])\)\s*',  # a), b), c)
            ],

            # BAB: berbagai format
            'bab': [
                r'(?:^|\n)\s*BAB\s+([IVX]+|[0-9]+)\s*(?:\n|$)',
                r'(?:^|\n)\s*Bab\s+([IVX]+|[0-9]+)\s*(?:\n|$)',
            ],

            # Bagian
            'bagian': [
                r'(?:^|\n)\s*BAGIAN\s+(.*?)(?:\n|$)',
                r'(?:^|\n)\s*Bagian\s+(.*?)(?:\n|$)',
            ],
        }

        # Cleaning patterns
        self.cleaning_patterns = [
            (r'\n\s*\n\s*\n+', '\n\n'),
            (r'[ \t]+', ' '),
            (r'\n[ \t]+', '\n'),
        ]

    def remove_unwanted_elements(self, text: str) -> str:
        """Menghapus elemen yang tidak diinginkan dari teks"""
        cleaned_text = text

        for category, patterns in self.removal_patterns.items():
            for pattern in patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)

        return cleaned_text

    def clean_text_formatting(self, text: str) -> str:
        """Membersihkan formatting teks"""
        cleaned = text

        for pattern, replacement in self.cleaning_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE)

        return cleaned.strip()

    def debug_print_sample(self, text: str, sample_size: int = 2000):
        """Print sample teks untuk debugging"""
        if self.debug_mode:
            logger.info("DEBUG: Sample text after cleaning:")
            logger.info("-" * 50)
            sample = text[:sample_size]
            logger.info(sample)
            logger.info("-" * 50)

    def test_patterns_on_text(self, text: str) -> Dict[str, List]:
        """Test semua pattern pada teks untuk debugging"""
        results = {}

        for pattern_type, patterns in self.structure_patterns.items():
            results[pattern_type] = []

            if isinstance(patterns, list):
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                    if matches:
                        results[pattern_type].extend(matches)
                        logger.info(f"Found {len(matches)} matches for {pattern_type}: {matches[:5]}...")
                        break
            else:
                matches = re.findall(patterns, text, re.MULTILINE | re.IGNORECASE)
                if matches:
                    results[pattern_type] = matches
                    logger.info(f"Found {len(matches)} matches for {pattern_type}: {matches[:5]}...")

        for pattern_type, matches in results.items():
            if not matches:
                logger.warning(f"No matches found for {pattern_type}")

        return results

    def extract_legal_structure(self, text: str) -> List[LegalElement]:
        """Ekstrak struktur legal dari teks dengan debugging"""

        # Debug: print sample
        self.debug_print_sample(text)

        # Debug: test patterns
        pattern_matches = self.test_patterns_on_text(text)

        elements = []
        lines = text.split('\n')

        current_pasal = None
        current_ayat = None
        current_bab = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Check untuk BAB dengan multiple patterns
            bab_found = False
            for pattern in self.structure_patterns['bab']:
                bab_match = re.match(pattern, line, re.IGNORECASE)
                if bab_match:
                    current_bab = bab_match.group(1)
                    bab_content = line.strip()

                    # Cek judul BAB di line berikutnya
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not re.match(r'(?:PASAL|Pasal|BAB|Bagian)', next_line):
                            bab_content += f": {next_line}"
                            i += 1

                    elements.append(LegalElement(
                        content=bab_content,
                        element_type='bab'
                    ))
                    logger.info(f"Found BAB: {bab_content}")
                    i += 1
                    bab_found = True
                    break

            if bab_found:
                continue

            # Check untuk Bagian dengan multiple patterns
            bagian_found = False
            for pattern in self.structure_patterns['bagian']:
                bagian_match = re.match(pattern, line, re.IGNORECASE)
                if bagian_match:
                    bagian_content = line.strip()
                    elements.append(LegalElement(
                        content=bagian_content,
                        element_type='bagian'
                    ))
                    logger.info(f"Found Bagian: {bagian_content}")
                    i += 1
                    bagian_found = True
                    break

            if bagian_found:
                continue

            # Check untuk Pasal dengan multiple patterns
            pasal_found = False
            for pattern in self.structure_patterns['pasal']:
                pasal_match = re.search(pattern, line, re.IGNORECASE)
                if pasal_match:
                    current_pasal = int(pasal_match.group(1))
                    current_ayat = None

                    # Kumpulkan konten pasal
                    content_lines = []
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            i += 1
                            continue

                        # Stop jika menemukan struktur baru
                        stop_found = False
                        for stop_pattern in (self.structure_patterns['pasal'] +
                                             self.structure_patterns['ayat'] +
                                             self.structure_patterns['bab']):
                            if re.search(stop_pattern, next_line, re.IGNORECASE):
                                stop_found = True
                                break

                        if stop_found:
                            break

                        content_lines.append(next_line)
                        i += 1

                    if content_lines:
                        content = ' '.join(content_lines)
                        elements.append(LegalElement(
                            pasal=current_pasal,
                            content=content,
                            element_type='pasal'
                        ))
                        logger.info(f"Found Pasal {current_pasal}: {content[:100]}...")

                    pasal_found = True
                    break

            if pasal_found:
                continue

            # Check untuk Ayat
            if current_pasal:
                ayat_found = False
                for pattern in self.structure_patterns['ayat']:
                    ayat_match = re.match(pattern, line)
                    if ayat_match:
                        current_ayat = int(ayat_match.group(1))

                        # Ambil sisa line sebagai konten
                        ayat_content = re.sub(pattern, '', line).strip()
                        content_lines = [ayat_content] if ayat_content else []

                        # Kumpulkan konten lanjutan
                        i += 1
                        while i < len(lines):
                            next_line = lines[i].strip()
                            if not next_line:
                                i += 1
                                continue

                            # Stop jika menemukan struktur baru
                            stop_found = False
                            for stop_pattern in (self.structure_patterns['pasal'] +
                                                 self.structure_patterns['ayat'] +
                                                 self.structure_patterns['poin'] +
                                                 self.structure_patterns['bab']):
                                if re.search(stop_pattern, next_line, re.IGNORECASE):
                                    stop_found = True
                                    break

                            if stop_found:
                                break

                            content_lines.append(next_line)
                            i += 1

                        if content_lines:
                            content = ' '.join(content_lines)
                            elements.append(LegalElement(
                                pasal=current_pasal,
                                ayat=current_ayat,
                                content=content,
                                element_type='ayat'
                            ))
                            logger.info(f"Found Ayat ({current_ayat}): {content[:50]}...")

                        ayat_found = True
                        break

                if ayat_found:
                    continue

            # Check untuk Poin
            if current_pasal:
                poin_found = False
                for pattern in self.structure_patterns['poin']:
                    poin_match = re.match(pattern, line)
                    if poin_match:
                        poin_letter = poin_match.group(1)
                        poin_content = re.sub(pattern, '', line).strip()

                        content_lines = [poin_content] if poin_content else []
                        i += 1
                        while i < len(lines):
                            next_line = lines[i].strip()
                            if not next_line:
                                i += 1
                                continue

                            # Stop jika menemukan struktur baru
                            stop_found = False
                            for stop_pattern in (self.structure_patterns['pasal'] +
                                                 self.structure_patterns['ayat'] +
                                                 self.structure_patterns['poin'] +
                                                 self.structure_patterns['bab']):
                                if re.search(stop_pattern, next_line, re.IGNORECASE):
                                    stop_found = True
                                    break

                            if stop_found:
                                break

                            content_lines.append(next_line)
                            i += 1

                        if content_lines:
                            content = ' '.join(content_lines)
                            elements.append(LegalElement(
                                pasal=current_pasal,
                                ayat=current_ayat,
                                poin=poin_letter,
                                content=content,
                                element_type='poin'
                            ))
                            logger.info(f"Found Poin {poin_letter}: {content[:50]}...")

                        poin_found = True
                        break

                if poin_found:
                    continue

            i += 1

        logger.info(f"Total elements extracted: {len(elements)}")
        return elements

    def create_hierarchical_text(self, elements: List[LegalElement]) -> List[Dict]:
        """Membuat teks hierarki untuk RAG dataset"""
        dataset = []

        for element in elements:
            identifier_parts = []
            if element.pasal:
                identifier_parts.append(f"Pasal {element.pasal}")
            if element.ayat:
                identifier_parts.append(f"ayat ({element.ayat})")
            if element.poin:
                identifier_parts.append(f"poin {element.poin}")

            identifier = " ".join(identifier_parts) if identifier_parts else element.element_type

            if identifier_parts:
                full_text = f"{identifier} berbunyi: {element.content}"
            else:
                full_text = element.content

            dataset.append({
                'identifier': identifier,
                'pasal': element.pasal,
                'ayat': element.ayat,
                'poin': element.poin,
                'element_type': element.element_type,
                'content': element.content,
                'full_text': full_text,
                'hierarchy_level': self._get_hierarchy_level(element)
            })

        return dataset

    def _get_hierarchy_level(self, element: LegalElement) -> int:
        """Menentukan level hierarki elemen"""
        if element.element_type == 'bab':
            return 1
        elif element.element_type == 'bagian':
            return 2
        elif element.element_type == 'pasal':
            return 3
        elif element.element_type == 'ayat':
            return 4
        elif element.element_type == 'poin':
            return 5
        else:
            return 6


class ImprovedBatchPDFProcessor:
    """Processor yang diperbaiki untuk memproses multiple PDF files dalam batch"""

    def __init__(self, input_folder: str = "data", output_folder: str = "processed"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.preprocessor = ImprovedLegalDocumentPreprocessor()

        # Create output folder structure
        self.setup_output_folders()

    def setup_output_folders(self):
        """Setup struktur folder output"""
        folders_to_create = [
            self.output_folder,
            self.output_folder / "json",
            self.output_folder / "csv",
            self.output_folder / "txt",
            self.output_folder / "combined",
            self.output_folder / "logs",
            self.output_folder / "debug"  # Tambah folder debug
        ]

        for folder in folders_to_create:
            folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output folders created in: {self.output_folder}")

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract teks dari file PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error reading page {page_num + 1} in {pdf_path.name}: {e}")
                        continue

            if not text.strip():
                logger.error(f"No text extracted from {pdf_path.name}")
                return None

            logger.info(f"Text extracted from {pdf_path.name}: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            return None

    def get_clean_filename(self, filename: str) -> str:
        """Generate clean filename untuk output"""
        name = Path(filename).stem
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        name = name.lower().strip('_')
        return name

    def process_single_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """Process single PDF file dengan debugging yang lebih baik"""
        logger.info(f"Processing: {pdf_path.name}")

        try:
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                return None

            clean_name = self.get_clean_filename(pdf_path.name)

            # Save raw text untuk debugging
            debug_raw_path = self.output_folder / "debug" / f"{clean_name}_raw.txt"
            with open(debug_raw_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)

            # Preprocess document
            logger.info(f"Preprocessing: {pdf_path.name}")
            cleaned_text = self.preprocessor.remove_unwanted_elements(raw_text)
            cleaned_text = self.preprocessor.clean_text_formatting(cleaned_text)

            # Save cleaned text untuk debugging
            debug_cleaned_path = self.output_folder / "debug" / f"{clean_name}_cleaned.txt"
            with open(debug_cleaned_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # Extract legal structure
            logger.info(f"Extracting legal structure: {pdf_path.name}")
            legal_elements = self.preprocessor.extract_legal_structure(cleaned_text)

            # Create hierarchical dataset
            dataset = self.preprocessor.create_hierarchical_text(legal_elements)

            # Prepare result
            result = {
                'document_name': pdf_path.stem,
                'source_file': pdf_path.name,
                'processed_date': datetime.now().isoformat(),
                'total_elements': len(dataset),
                'cleaned_text': cleaned_text,
                'structured_data': dataset,
                'statistics': self._generate_statistics(dataset)
            }

            # Save individual files
            self._save_individual_files(result, clean_name)

            logger.info(f"Successfully processed: {pdf_path.name} ({len(dataset)} elements)")
            return result

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return None

    def _generate_statistics(self, dataset: List[Dict]) -> Dict:
        """Generate statistik dari dataset"""
        stats = {
            'total_elements': len(dataset),
            'by_type': {},
            'by_hierarchy_level': {},
            'pasal_count': 0,
            'ayat_count': 0,
            'poin_count': 0
        }

        pasal_numbers = set()

        for item in dataset:
            element_type = item['element_type']
            stats['by_type'][element_type] = stats['by_type'].get(element_type, 0) + 1

            level = item['hierarchy_level']
            stats['by_hierarchy_level'][level] = stats['by_hierarchy_level'].get(level, 0) + 1

            if item['pasal']:
                pasal_numbers.add(item['pasal'])
            if item['ayat']:
                stats['ayat_count'] += 1
            if item['poin']:
                stats['poin_count'] += 1

        stats['pasal_count'] = len(pasal_numbers)
        stats['unique_pasal_numbers'] = sorted(list(pasal_numbers))

        return stats

    def _save_individual_files(self, result: Dict, clean_name: str):
        """Save individual processed files"""

        # Save JSON
        json_path = self.output_folder / "json" / f"{clean_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Save CSV
        if result['structured_data']:
            csv_path = self.output_folder / "csv" / f"{clean_name}.csv"
            df = pd.DataFrame(result['structured_data'])
            df['source_document'] = result['document_name']
            df['source_file'] = result['source_file']
            df.to_csv(csv_path, index=False, encoding='utf-8')

        # Save TXT (hierarchy)
        txt_path = self.output_folder / "txt" / f"{clean_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"DOKUMEN: {result['document_name']}\n")
            f.write(f"SOURCE: {result['source_file']}\n")
            f.write(f"PROCESSED: {result['processed_date']}\n")
            f.write(f"TOTAL ELEMENTS: {result['total_elements']}\n")
            f.write("=" * 80 + "\n\n")

            if result['structured_data']:
                for item in result['structured_data']:
                    f.write(f"{item['full_text']}\n\n")
            else:
                f.write("No structured elements found.\n")

    def process_all_pdfs(self) -> Dict:
        """Process all PDFs in input folder"""
        logger.info(f"Starting batch processing from: {self.input_folder}")

        pdf_files = list(self.input_folder.glob("*.pdf"))

        if not pdf_files:
            logger.error(f"No PDF files found in {self.input_folder}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        processed_results = []
        failed_files = []

        for pdf_file in pdf_files:
            result = self.process_single_pdf(pdf_file)
            if result:
                processed_results.append(result)
            else:
                failed_files.append(pdf_file.name)

        # Create combined datasets
        logger.info("Creating combined datasets...")
        combined_result = self._create_combined_datasets(processed_results)

        # Generate summary report
        summary = self._generate_summary_report(processed_results, failed_files)

        logger.info(f"Batch processing completed!")
        logger.info(f"Successfully processed: {len(processed_results)} files")
        logger.info(f"Failed files: {len(failed_files)}")

        return {
            'processed_results': processed_results,
            'failed_files': failed_files,
            'combined_result': combined_result,
            'summary': summary
        }

    def _create_combined_datasets(self, results: List[Dict]) -> Dict:
        """Create combined datasets from all processed files"""

        all_data = []
        for result in results:
            for item in result['structured_data']:
                item['source_document'] = result['document_name']
                item['source_file'] = result['source_file']
                all_data.append(item)

        combined_json_path = self.output_folder / "combined" / "all_legal_documents.json"
        combined_result = {
            'combined_date': datetime.now().isoformat(),
            'total_documents': len(results),
            'total_elements': len(all_data),
            'documents': [r['document_name'] for r in results],
            'data': all_data
        }

        with open(combined_json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)

        if all_data:
            combined_csv_path = self.output_folder / "combined" / "all_legal_documents.csv"
            df = pd.DataFrame(all_data)
            df.to_csv(combined_csv_path, index=False, encoding='utf-8')

            # Save RAG-ready format
            rag_data = []
            for item in all_data:
                rag_entry = {
                    'id': f"{item.get('source_document', 'unknown')}_{item.get('pasal', 'x')}_{item.get('ayat', 'x')}_{item.get('poin', 'x')}",
                    'content': item['full_text'],
                    'metadata': {
                        'source_document': item.get('source_document'),
                        'source_file': item.get('source_file'),
                        'pasal': item.get('pasal'),
                        'ayat': item.get('ayat'),
                        'poin': item.get('poin'),
                        'element_type': item.get('element_type'),
                        'hierarchy_level': item.get('hierarchy_level')
                    }
                }
                rag_data.append(rag_entry)

            rag_json_path = self.output_folder / "combined" / "rag_dataset.json"
            with open(rag_json_path, 'w', encoding='utf-8') as f:
                json.dump(rag_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Combined datasets saved:")
            logger.info(f"   - JSON: {combined_json_path}")
            logger.info(f"   - CSV: {combined_csv_path}")
            logger.info(f"   - RAG: {rag_json_path}")

        return combined_result

    def _generate_summary_report(self, results: List[Dict], failed_files: List[str]) -> Dict:
        """Generate summary report"""

        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_pdf_files': len(results) + len(failed_files),
            'successfully_processed': len(results),
            'failed_files': len(failed_files),
            'failed_file_names': failed_files,
            'total_elements_extracted': sum(r['total_elements'] for r in results),
            'documents_processed': [],
            'overall_statistics': {
                'total_pasal': 0,
                'total_ayat': 0,
                'total_poin': 0,
                'by_document_type': {}
            }
        }

        for result in results:
            doc_info = {
                'document_name': result['document_name'],
                'source_file': result['source_file'],
                'elements_count': result['total_elements'],
                'statistics': result['statistics']
            }
            summary['documents_processed'].append(doc_info)

            # Aggregate statistics
            summary['overall_statistics']['total_pasal'] += result['statistics'].get('pasal_count', 0)
            summary['overall_statistics']['total_ayat'] += result['statistics'].get('ayat_count', 0)
            summary['overall_statistics']['total_poin'] += result['statistics'].get('poin_count', 0)

        # Save summary report
        summary_path = self.output_folder / "logs" / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Save readable summary
        readable_summary_path = self.output_folder / "logs" / "processing_summary.txt"
        with open(readable_summary_path, 'w', encoding='utf-8') as f:
            f.write("LEGAL DOCUMENTS PREPROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {summary['processing_date']}\n")
            f.write(f"Total PDF Files: {summary['total_pdf_files']}\n")
            f.write(f"Successfully Processed: {summary['successfully_processed']}\n")
            f.write(f"Failed Files: {summary['failed_files']}\n")
            f.write(f"Total Elements Extracted: {summary['total_elements_extracted']}\n\n")

            if failed_files:
                f.write("FAILED FILES:\n")
                for file in failed_files:
                    f.write(f"  - {file}\n")
                f.write("\n")

            f.write("PROCESSED DOCUMENTS:\n")
            for doc in summary['documents_processed']:
                f.write(f"  - {doc['document_name']}: {doc['elements_count']} elements\n")
                if doc['statistics']['by_type']:
                    f.write(f"    Types: {doc['statistics']['by_type']}\n")

        logger.info(f"Summary report saved: {summary_path}")
        return summary


def process_legal_documents_improved(input_folder: str = "data", output_folder: str = "processed"):
    """
    Args:
        input_folder: Folder yang berisi file PDF
        output_folder: Folder output untuk hasil preprocessing

    Returns:
        Dict berisi hasil processing
    """

    processor = ImprovedBatchPDFProcessor(input_folder, output_folder)
    result = processor.process_all_pdfs()
    return result


# Fungsi utilitas untuk analisis manual
def analyze_single_document(pdf_path: str, output_folder: str = "analysis"):
    """
    Fungsi untuk menganalisis satu dokumen secara detail
    """
    from pathlib import Path

    # Setup
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    processor = ImprovedBatchPDFProcessor(".", output_folder)

    # Process single file
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"File tidak ditemukan: {pdf_path}")
        return

    result = processor.process_single_pdf(pdf_file)

    if result:
        print(f"Berhasil memproses: {pdf_file.name}")
        print(f"Total elemen: {result['total_elements']}")
        print(f"Statistik: {result['statistics']}")

        # Save detailed analysis
        analysis_path = output_path / f"analysis_{processor.get_clean_filename(pdf_file.name)}.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Analisis disimpan di: {analysis_path}")
    else:
        print(f"Gagal memproses: {pdf_file.name}")


# Example usage dan testing
if __name__ == "__main__":
    print("IMPROVED LEGAL DOCUMENTS BATCH PROCESSOR")
    print("=" * 60)

    # Check if data folder exists
    if not Path("data").exists():
        print("Folder 'data' tidak ditemukan!")
        print("Buat folder 'data' dan letakkan file PDF di dalamnya")

        # Option to analyze single file
        import sys

        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
            print(f"Menganalisis file tunggal: {pdf_path}")
            analyze_single_document(pdf_path)

        exit(1)

    try:
        # Process all documents with improved algorithm
        result = process_legal_documents_improved(
            input_folder="data",
            output_folder="processed_improved"
        )

        print("\nPROCESSING COMPLETED!")
        print(f"Successfully processed: {len(result['processed_results'])} documents")
        print(f"Failed: {len(result['failed_files'])} documents")
        print(f"Total elements: {result['combined_result']['total_elements']}")

        print("\nOutput Structure:")
        print("processed_improved/")
        print("├── json/          # Individual JSON files")
        print("├── csv/           # Individual CSV files")
        print("├── txt/           # Individual text files")
        print("├── debug/         # Raw & cleaned text for debugging")
        print("├── combined/      # Combined datasets")
        print("│   ├── all_legal_documents.json")
        print("│   ├── all_legal_documents.csv")
        print("│   └── rag_dataset.json")
        print("└── logs/          # Processing logs and summary")

        # Show statistics by document
        print("\nPROCESSING SUMMARY:")
        for doc_result in result['processed_results']:
            stats = doc_result['statistics']
            print(f"{doc_result['document_name']}: {doc_result['total_elements']} elements")
            if stats['by_type']:
                types_str = ", ".join([f"{k}={v}" for k, v in stats['by_type'].items()])
                print(f"   └─ Types: {types_str}")

        if result['failed_files']:
            print(f"\nFailed files: {', '.join(result['failed_files'])}")
            print("Check debug files in processed_improved/debug/ folder")

        print("\nDEBUGGING TIPS:")
        print("1. Check files in debug/ folder to see raw and cleaned text")
        print("2. Look at preprocessing_log.txt for detailed processing info")
        print("3. If no elements found, patterns may need further adjustment")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Fatal error: {e}")

        # Print debugging info
        print("\nDEBUGGING INFO:")
        print("1. Check that PDF files are readable")
        print("2. Ensure documents contain Indonesian legal structure")
        print("3. Review preprocessing_log.txt for detailed errors")

        import traceback

        traceback.print_exc()