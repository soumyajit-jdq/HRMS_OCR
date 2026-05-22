import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from service import ProcessingService
from models import (
    AadharCard, Pancard, Passport, BankPassbook, CancelCheck, BirthCertificate
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def process_document(file: UploadFile, extractor_func):
    """
    Generic pipeline: Validate -> OCR -> Extract
    """
    try:
        file_bytes = await file.read()
        
        # 1. Automatic Validation
        validation = await ProcessingService.validate_document(file_bytes, file.filename)
        if not validation.is_valid:
            logger.warning(f"Quality Check Failed: {validation.instruction}")
            raise HTTPException(status_code=400, detail=validation.instruction)
            
        # 2. Extract Text via OCR
        # Handle PDFs by converting the first page to an image
        if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
            images = await ProcessingService.process_pdf_pages(file_bytes)
            if not images:
                raise HTTPException(status_code=400, detail="Could not extract images from PDF")
            ocr_text = await ProcessingService.run_ocr(images[0])
        else:
            ocr_text = await ProcessingService.run_ocr(file_bytes)
            
        if ocr_text.startswith("OCR Failed"):
            raise HTTPException(status_code=500, detail=ocr_text)
            
        logger.info(f"--- RAW OCR TEXT START ---\n{ocr_text}\n--- RAW OCR TEXT END ---")
        
        # 3. Apply Regex Extraction
        try:
            structured_data = extractor_func(ocr_text)
            return structured_data
        except ValueError as ve:
            logger.warning(f"Classification / Extraction Error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Extraction route failed")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import ValidationError

@router.post("/aadhar", response_model=AadharCard)
async def extract_aadhar(file: UploadFile = File(...)):
    """Upload Aadhar Card for Extraction"""
    result = await process_document(file, ProcessingService.extract_aadhar)
    try:
        return AadharCard(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")

@router.post("/pan", response_model=Pancard)
async def extract_pan(file: UploadFile = File(...)):
    """Upload PAN Card for Extraction"""
    result = await process_document(file, ProcessingService.extract_pan)
    try:
        return Pancard(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")

@router.post("/passport", response_model=Passport)
async def extract_passport(file: UploadFile = File(...)):
    """Upload Passport for Extraction"""
    result = await process_document(file, ProcessingService.extract_passport)
    try:
        return Passport(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")

@router.post("/bank_passbook", response_model=BankPassbook)
async def extract_bank_passbook(file: UploadFile = File(...)):
    """Upload Bank Passbook for Extraction"""
    result = await process_document(file, ProcessingService.extract_bank_passbook)
    try:
        return BankPassbook(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")

@router.post("/cancel_check", response_model=CancelCheck)
async def extract_cancel_check(file: UploadFile = File(...)):
    """Upload Cancelled Check for Extraction"""
    result = await process_document(file, ProcessingService.extract_cancel_check)
    try:
        return CancelCheck(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")

@router.post("/birth_certificate", response_model=BirthCertificate)
async def extract_birth_certificate(file: UploadFile = File(...)):
    """Upload Birth Certificate for Extraction"""
    result = await process_document(file, ProcessingService.extract_birth_certificate)
    try:
        return BirthCertificate(**result)
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
        raise HTTPException(status_code=400, detail=f"Failed to extract mandatory fields from the document. Missing: {', '.join(missing_fields)}")
