"""Centralized column/schema expectations for all raw datasets.

Verified against the actual files on disk with pandas (see docs/data_contract.md).
Do not duplicate these column lists elsewhere in the project.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Motor: shivamb/vehicle-claim-fraud-detection (fraud_oracle.csv)
# ---------------------------------------------------------------------------

MOTOR_RAW_PATH = DATA_RAW_DIR / "motor" / "fraud_oracle.csv"

MOTOR_TARGET = "FraudFound_P"

# Identifiers: kept for traceability/joins, excluded from the model feature matrix.
MOTOR_ID_COLUMNS = ["PolicyNumber", "RepNumber"]

MOTOR_NUMERICAL_COLUMNS = [
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Age",
    "Deductible",
    "DriverRating",
    "Year",
]

MOTOR_CATEGORICAL_COLUMNS = [
    "Month",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "BasePolicy",
]

MOTOR_ALL_COLUMNS = (
    MOTOR_ID_COLUMNS + [MOTOR_TARGET] + MOTOR_NUMERICAL_COLUMNS + MOTOR_CATEGORICAL_COLUMNS
)

# ---------------------------------------------------------------------------
# Health: rohitrox/healthcare-provider-fraud-detection-analysis
# ---------------------------------------------------------------------------

HEALTH_RAW_DIR = DATA_RAW_DIR / "health"

HEALTH_TRAIN_MAIN_PATH = HEALTH_RAW_DIR / "Train-1542865627584.csv"
HEALTH_TRAIN_BENEFICIARY_PATH = HEALTH_RAW_DIR / "Train_Beneficiarydata-1542865627584.csv"
HEALTH_TRAIN_INPATIENT_PATH = HEALTH_RAW_DIR / "Train_Inpatientdata-1542865627584.csv"
HEALTH_TRAIN_OUTPATIENT_PATH = HEALTH_RAW_DIR / "Train_Outpatientdata-1542865627584.csv"

# Kaggle's Test* files carry no PotentialFraud label at all (verified) and are
# out of scope for supervised training/evaluation in this project — see
# docs/data_contract.md section 3. Kept only for potential future unlabeled use.
HEALTH_TEST_MAIN_PATH = HEALTH_RAW_DIR / "Test-1542969243754.csv"
HEALTH_TEST_BENEFICIARY_PATH = HEALTH_RAW_DIR / "Test_Beneficiarydata-1542969243754.csv"
HEALTH_TEST_INPATIENT_PATH = HEALTH_RAW_DIR / "Test_Inpatientdata-1542969243754.csv"
HEALTH_TEST_OUTPATIENT_PATH = HEALTH_RAW_DIR / "Test_Outpatientdata-1542969243754.csv"

# Present on disk but NOT the locked health dataset: has no fraud/label column
# of any kind (leandrenash/enhanced-health-insurance-claims-dataset). Must
# never be loaded as a fraud data source.
HEALTH_UNUSED_NO_LABEL_PATH = HEALTH_RAW_DIR / "enhanced_health_insurance_claims.csv"

HEALTH_PROVIDER_COL = "Provider"
HEALTH_PROVIDER_LABEL_COL = "PotentialFraud"  # provider-level, Yes/No
HEALTH_BENE_ID_COL = "BeneID"
HEALTH_CLAIM_ID_COL = "ClaimID"

HEALTH_CLAIM_SHARED_COLUMNS = [
    HEALTH_BENE_ID_COL,
    HEALTH_CLAIM_ID_COL,
    "ClaimStartDt",
    "ClaimEndDt",
    HEALTH_PROVIDER_COL,
    "InscClaimAmtReimbursed",
    "AttendingPhysician",
    "OperatingPhysician",
    "OtherPhysician",
    "DeductibleAmtPaid",
    "ClmAdmitDiagnosisCode",
]

HEALTH_INPATIENT_ONLY_COLUMNS = ["AdmissionDt", "DischargeDt", "DiagnosisGroupCode"]

HEALTH_DIAGNOSIS_CODE_COLUMNS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
HEALTH_PROCEDURE_CODE_COLUMNS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

HEALTH_BENEFICIARY_CHRONIC_COLUMNS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]

HEALTH_BENEFICIARY_COLUMNS = [
    HEALTH_BENE_ID_COL,
    "DOB",
    "DOD",
    "Gender",
    "Race",
    "RenalDiseaseIndicator",
    "State",
    "County",
    "NoOfMonths_PartACov",
    "NoOfMonths_PartBCov",
    *HEALTH_BENEFICIARY_CHRONIC_COLUMNS,
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
]

# Name used for the claim-level target inherited from the provider-level
# PotentialFraud label. Must never be shortened to a bare "fraud" column —
# see docs/data_contract.md section 11.
HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL = "claim_fraud_label_provider_derived"

# ---------------------------------------------------------------------------
# Demographics reference: patelparth3399/indian-insurance-premium-prediction-dataset
# ---------------------------------------------------------------------------

DEMOGRAPHICS_RAW_PATH = DATA_RAW_DIR / "demographics" / "Indian_Insurance_Data.csv"

DEMOGRAPHICS_COLUMNS = [
    "age",
    "weight",
    "height",
    "income_lpa",
    "smoker",
    "city",
    "occupation",
    "insurance_premium_category",
]
