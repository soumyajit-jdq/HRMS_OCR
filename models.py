from pydantic import BaseModel, Field, ConfigDict, field_validator

class ValidationResponse(BaseModel):
    is_valid: bool
    instruction: str
    file_type: str

class AadharCard(BaseModel):
    aadhar_number: str
    name: str
    dob: str
    gender: str

class Pancard(BaseModel):
    pan_number: str
    name: str
    fathers_name: str
    dob: str

class Passport(BaseModel):
    passport_number: str
    name: str
    surname: str
    dob: str
    date_of_issue: str
    date_of_expiry: str

class BankPassbook(BaseModel):
    customer_name: str
    cif_no: str
    account_number: str
    branch_code: str
    bank_name: str
    ifsc_code: str
    micr: str

class CancelCheck(BaseModel):
    account_number: str
    ifsc_code: str
    check_number: str

class BirthCertificate(BaseModel):
    name: str
    gender: str
    dob: str
    father: str
    mother: str
    place_of_birth: str
    registration_number: str
    registration_date: str
    