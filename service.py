import os
import re
import anyio
import io
import logging
from PIL import Image
import httpx
from dotenv import load_dotenv
from preprocessing import validate_image_quality
from models import ValidationResponse, AadharCard, Pancard, Passport, BankPassbook, CancelCheck, BirthCertificate

load_dotenv()

# Setup Logging
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEYS = [k.strip() for k in os.getenv("OCR_API_KEY", "").split(",") if k.strip()]

class ProcessingService:
    @staticmethod
    async def validate_document(file_bytes: bytes, filename: str) -> ValidationResponse:
        """Runs the preprocessing quality checks in a separate thread to avoid blocking."""
        def sync_validate():
            is_valid, msg = validate_image_quality(file_bytes, filename)
            file_type = "PDF" if file_bytes.startswith(b"%PDF") else "Image"
            return is_valid, msg, file_type
            
        is_valid, msg, file_type = await anyio.to_thread.run_sync(sync_validate)
        return ValidationResponse(is_valid=is_valid, instruction=msg, file_type=file_type)

    @staticmethod
    async def compress_image(image_bytes: bytes, max_kb: int = 1000):
        """High-resolution compression for OCR.space (1MB limit)."""
        def sync_compress():
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # If already small enough, don't touch it
            if len(image_bytes) <= max_kb * 1024:
                return image_bytes
                
            # Try to save with high quality first
            quality = 90
            buffer = io.BytesIO()
            while quality > 10:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                if len(buffer.getvalue()) <= max_kb * 1024:
                    logger.info(f"Image compressed to {len(buffer.getvalue())//1024}KB at quality {quality}")
                    return buffer.getvalue()
                quality -= 10
            img.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=20)
            return buffer.getvalue()
            
        return await anyio.to_thread.run_sync(sync_compress)

    @staticmethod
    async def process_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
        import fitz
        def sync_pdf_process():
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = min(len(doc), max_pages)
                if num_pages == 0:
                    return []
                
                all_images = []
                for i in range(num_pages):
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    all_images.append(pix.tobytes("jpg"))
                doc.close()
                return all_images
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return []
        
        return await anyio.to_thread.run_sync(sync_pdf_process)

    @staticmethod
    async def run_ocr(image_bytes: bytes):
        """Asynchronous call to OCR.space API with multi-key rotation."""
        if not OCR_API_KEYS:
            return "OCR Failed: No API keys configured"

        compressed_bytes = await ProcessingService.compress_image(image_bytes)
        url = "https://api.ocr.space/parse/image"
        
        async def perform_request(client, api_key):
            files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
            # isTable=False forces natural top-to-bottom reading order, better for ID cards
            data = {"apikey": api_key, "language": "eng", "isTable": False, "OCREngine": 2}
            logger.info(f"Attempting OCR with key: {api_key[:5]}...")
            return await client.post(url, files=files, data=data, timeout=60)

        last_error = "Unknown Error"
        for api_key in OCR_API_KEYS:
            try:
                if hasattr(ProcessingService, '_shared_client') and ProcessingService._shared_client:
                    response = await perform_request(ProcessingService._shared_client, api_key)
                else:
                    async with httpx.AsyncClient() as client:
                        response = await perform_request(client, api_key)
                
                if response.status_code == 403:
                    logger.warning(f"OCR Key {api_key[:5]} returned 403 Forbidden. Trying next key...")
                    last_error = "403 Forbidden"
                    continue

                result = response.json()
                if result.get("OCRExitCode") != 1:
                    err = result.get('ErrorMessage')
                    logger.warning(f"OCR Key {api_key[:5]} failed: {err}")
                    last_error = err
                    continue
                    
                return result["ParsedResults"][0]["ParsedText"]
            except Exception as e:
                logger.error(f"OCR Error with key {api_key[:5]}: {e}")
                last_error = str(e)
                continue
        
        return f"OCR Failed: {last_error}"

    # --- REGEX EXTRACTORS ---

    @staticmethod
    def extract_aadhar(text: str) -> dict:
        data = {}
        text_lower = text.lower()
        if "government of india" not in text_lower and "aadhar" not in text_lower and "uidai" not in text_lower:
            raise ValueError("Document does not appear to be an Aadhar Card.")
            
        aadhar_match = re.search(r'\b\d{4}[ \t]*\d{4}[ \t]*\d{4}\b', text)
        if aadhar_match:
            data['aadhar_number'] = aadhar_match.group().replace(" ", "").replace("\t", "")
            
        dob_match = re.search(r'(?:DOB|Year of Birth|YOB|DOB/YOB)[\s:]*(\d{2}/\d{2}/\d{4}|\d{4})', text, re.IGNORECASE)
        if dob_match:
            data['dob'] = dob_match.group(1)
            
        gender_match = re.search(r'\b(Male|Female|Transgender)\b', text, re.IGNORECASE)
        if gender_match:
            data['gender'] = gender_match.group(1)
            
        # Heuristic for name: Look for the line above DOB
        tokens = [t.strip() for t in re.split(r'[\t\n]+', text) if t.strip()]
        for i, token in enumerate(tokens):
            if re.search(r'(?:DOB|Year of Birth|YOB)', token, re.IGNORECASE) and i > 0:
                clean_name = re.sub(r'[^A-Za-z\s\.]', '', tokens[i-1]).strip()
                if clean_name:
                    data['name'] = clean_name
                break
                
        return data

    @staticmethod
    def extract_pan(text: str) -> dict:
        data = {}
        text_lower = text.lower()
        if "income tax department" not in text_lower and "permanent account number" not in text_lower:
            raise ValueError("Document does not appear to be a PAN Card.")
            
        pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', text)
        if pan_match:
            data['pan_number'] = pan_match.group()
            
        dob_match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
        if dob_match:
            data['dob'] = dob_match.group(1)
            
        tokens = [t.strip() for t in re.split(r'[\t\n]+', text) if t.strip()]
        
        # 1. Try Anchor-based Extraction
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            if not data.get('name') and re.search(r'\bname\b', token_lower) and "father" not in token_lower:
                if i + 1 < len(tokens):
                    clean_name = re.sub(r'[^A-Za-z\s\.]', '', tokens[i+1]).strip()
                    if clean_name and not re.search(r'\d', tokens[i+1]):
                        data['name'] = clean_name.upper()
                        
            if not data.get('fathers_name') and "father" in token_lower:
                if i + 1 < len(tokens):
                    clean_father = re.sub(r'[^A-Za-z\s\.]', '', tokens[i+1]).strip()
                    if clean_father and not re.search(r'\d', tokens[i+1]):
                        data['fathers_name'] = clean_father.upper()
        
        # 2. Fallback Positional Extraction
        if not data.get('name') or not data.get('fathers_name'):
            candidates = []
            ignore_words = ["income tax", "govt", "india", "permanent", "account", "number", "card", "signature", "sgnature", "sign", "name", "father", "department", "dob", "birth", "year", "date", "issue"]
            
            for token in tokens:
                if not token: continue
                if pan_match and pan_match.group() in token: continue
                if dob_match and dob_match.group() in token: continue
                if re.search(r'\d', token): continue
                if any(iw in token.lower() for iw in ignore_words): continue
                
                clean_token = re.sub(r'[^A-Za-z\s\.]', '', token).strip()
                if len(clean_token) > 4:  # Avoid short artifacts like 'RRAR'
                    candidates.append(clean_token)
                    
            if not data.get('name') and len(candidates) >= 1:
                data['name'] = candidates[0].upper()
            if not data.get('fathers_name') and len(candidates) >= 2:
                data['fathers_name'] = candidates[1].upper()
                
        return data

    @staticmethod
    def extract_passport(text: str) -> dict:
        data = {}
        text_lower = text.lower()
        if "passport" not in text_lower and "republic of india" not in text_lower:
            raise ValueError("Document does not appear to be a Passport.")
            
        passport_match = re.search(r'\b[A-Z]{1}[0-9]{7}\b', text)
        if passport_match:
            data['passport_number'] = passport_match.group()
            
        dates = re.findall(r'\b(\d{2}/\d{2}/\d{4})\b', text)
        if len(dates) >= 1:
            data['dob'] = dates[0]
        if len(dates) >= 2:
            data['date_of_issue'] = dates[1]
        if len(dates) >= 3:
            data['date_of_expiry'] = dates[2]
            
        # Extract Name using MRZ (Machine Readable Zone)
        # Require '<<' and at least 10 uppercase/< characters to avoid matching random words like 'PUBLIC'
        mrz_match = re.search(r'P[A-Z<0-9]{4}([A-Z<]+<<[A-Z<]+)', text)
        if mrz_match:
            mrz_name_part = mrz_match.group(1).strip('<')
            parts = mrz_name_part.split('<<')
            if len(parts) >= 2:
                data['surname'] = parts[0].replace('<', ' ').strip()
                data['name'] = parts[1].replace('<', ' ').strip() # Given name
            else:
                data['name'] = parts[0].replace('<', ' ').strip()
                data['surname'] = ""
                
        # Fallback to anchors if MRZ fails
        if not data.get('name') and not data.get('surname'):
            tokens = [t.strip() for t in re.split(r'[\t\n]+', text) if t.strip()]
            surname = ""
            given_name = ""
            for i, token in enumerate(tokens):
                if "surname" in token.lower() and i + 1 < len(tokens):
                    surname = re.sub(r'[^A-Z\s]', '', tokens[i+1].upper()).strip()
                if "given name" in token.lower() and i + 1 < len(tokens):
                    given_name = re.sub(r'[^A-Z\s]', '', tokens[i+1].upper()).strip()
            
            if surname:
                data['surname'] = surname
            if given_name:
                data['name'] = given_name
            elif surname and not given_name:
                data['name'] = surname  # fallback if given name is empty
                
        # Ensure mandatory fields have at least an empty string to prevent validation crash if partially found
        if 'name' not in data:
            data['name'] = ""
        if 'surname' not in data:
            data['surname'] = ""
            
        return data

    @staticmethod
    def extract_bank_passbook(text: str) -> dict:
        data = {}
        
        ac_match = re.search(r'(?:A/c No|Account No|Account Number)[\s:\.]*(\d{9,18})', text, re.IGNORECASE)
        if ac_match:
            data['account_number'] = ac_match.group(1)
            
        ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
        if ifsc_match:
            data['ifsc_code'] = ifsc_match.group()
            
        return data

    @staticmethod
    def extract_cancel_check(text: str) -> dict:
        data = {}
        text_lower = text.lower()
        
        if "cancelled" not in text_lower and "cancel" not in text_lower:
            pass # Keep going but it's suspicious
            
        ac_match = re.search(r'(?:A/c No|Account No|Account Number)[\s:\.]*(\d{9,18})', text, re.IGNORECASE)
        if ac_match:
            data['account_number'] = ac_match.group(1)
            
        ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
        if ifsc_match:
            data['ifsc_code'] = ifsc_match.group()
            
        # The check number in India is a 6 digit code at the bottom
        check_num_match = re.search(r'(?<!\d)(\d{6})(?!\d)', text)
        if check_num_match:
            data['check_number'] = check_num_match.group(1)
        else:
            data['check_number'] = ""
            
        return data

    @staticmethod
    def extract_birth_certificate(text: str) -> dict:
        data = {}
        text_lower = text.lower()
        if "birth" not in text_lower and "birth certificate" not in text_lower:
            raise ValueError("Document does not appear to be a Birth Certificate.")
            
        label_map = {
            "name": "name", "sex": "gender", "ser": "gender", "gender": "gender",
            "date of birth": "dob", "place of birth": "place_of_birth",
            "name of father": "father", "father's name": "father",
            "name of mother": "mother", "mother's name": "mother",
            "registration no": "registration_number", "registration number": "registration_number",
            "date of registration": "registration_date"
        }
        
        def is_label(line: str):
            lower = line.lower().strip(' :.-')
            if lower in label_map:
                return label_map[lower]
            for key in sorted(label_map.keys(), key=len, reverse=True):
                if lower.startswith(key):
                    return label_map[key]
            return None
            
        ignore_phrases = [
            "municipal corporation", "health department", "birth certificate", "issued under section",
            "this is to certify", "signature", "date:", "form no", "register for", "hospital of district",
            "state west bengal", "local area", "of district", "act"
        ]
        
        queue = []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            
            if ":" in line and not line.startswith(":"):
                parts = line.split(":", 1)
                lbl = is_label(parts[0])
                if lbl and parts[1].strip():
                    data[lbl] = parts[1].strip()
                    continue
                    
            lower = line.lower()
            if any(ign in lower for ign in ignore_phrases) and not line.startswith(":"):
                continue
                
            lbl = is_label(line)
            if lbl:
                queue.append(lbl)
            else:
                val = line.lstrip(' :').strip()
                if val and queue:
                    target = queue.pop(0)
                    if target not in data:
                        data[target] = val
                        
        # Clean dob from queue if possible
        if data.get('dob'):
            m = re.search(r'\b(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})\b', data['dob'])
            if m: data['dob'] = m.group(1)
            
        # If dob wasn't found in queue, fallback to the first date in text
        dates = re.findall(r'\b(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})\b', text)
        if not data.get('dob') and dates:
            data['dob'] = dates[0]
            
        # Clean registration_date from queue if possible
        if data.get('registration_date'):
            m = re.search(r'\b(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})\b', data['registration_date'])
            if m: data['registration_date'] = m.group(1)
            
        # Smart Registration Date Extraction: Look for dates immediately following the label
        reg_date_match = re.search(r'(?:Date of Registration|Registration Date)[\s:\.]*(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})', text, re.IGNORECASE)
        if reg_date_match:
            data['registration_date'] = reg_date_match.group(1)
            
        reg_match = re.search(r'(?:Registration No|Reg No|Registration Number)[ \t:\.]*([A-Z0-9/-]+)', text, re.IGNORECASE)
        if reg_match:
            data['registration_number'] = reg_match.group(1)
            
        # Safely mirror fields so it works with ANY pydantic model the user types
        if 'father' in data: data['fathers_name'] = data['father']
        if 'fathers_name' in data: data['father'] = data['fathers_name']
        if 'mother' in data: data['mothers_name'] = data['mother']
        if 'mothers_name' in data: data['mother'] = data['mothers_name']
            
        for field in ["name", "gender", "dob", "father", "fathers_name", "mother", "mothers_name", "place_of_birth", "registration_number", "registration_date"]:
            if field not in data:
                data[field] = ""
                
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = v.rstrip(':').strip()
                
        return data
