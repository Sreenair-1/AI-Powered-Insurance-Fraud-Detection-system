import os
from pathlib import Path
import pandas as pd

def setup_test_datasets():
    """Create minimal mock datasets for CI testing."""
    base_path = Path("data/raw")
    
    # Create directories
    (base_path / "demographics").mkdir(parents=True, exist_ok=True)
    (base_path / "motor").mkdir(parents=True, exist_ok=True)
    (base_path / "health").mkdir(parents=True, exist_ok=True)
    
    # Create minimal motor dataset
    motor_df = pd.DataFrame({
        "PolicyNumber": [1, 2, 3],
        "RepNumber": [1, 2, 3],
        "FraudFound_P": [0, 1, 0],
        "WeekOfMonth": [1, 2, 3],
        "WeekOfMonthClaimed": [1, 2, 3],
        "Age": [25, 35, 45],
        "Deductible": [500, 1000, 1500],
        "DriverRating": [1, 2, 3],
        "Year": [2020, 2021, 2022],
        "Month": [1, 2, 3],
        "DayOfWeek": [1, 2, 3],
        "Make": ["Honda", "Toyota", "Ford"],
        "AccidentArea": ["Urban", "Rural", "Urban"],
        "DayOfWeekClaimed": [1, 2, 3],
        "MonthClaimed": [1, 2, 3],
        "Sex": ["M", "F", "M"],
        "MaritalStatus": ["Single", "Married", "Single"],
        "Fault": ["Yes", "No", "Yes"],
        "PolicyType": ["Comprehensive", "Third Party", "Comprehensive"],
        "VehicleCategory": ["2-Door", "4-Door", "2-Door"],
        "VehiclePrice": ["20000", "30000", "40000"],
        "Days_Policy_Accident": [1, 2, 3],
        "Days_Policy_Claim": [1, 2, 3],
        "PastNumberOfClaims": [0, 1, 2],
        "AgeOfVehicle": [5, 10, 15],
        "AgeOfPolicyHolder": [1, 2, 3],
        "PoliceReportFiled": ["Yes", "No", "Yes"],
        "WitnessPresent": ["Yes", "No", "Yes"],
        "AgentType": ["Agent", "Broker", "Agent"],
        "NumberOfSuppliments": [1, 2, 3],
        "AddressChange_Claim": [1, 2, 3],
        "NumberOfCars": [1, 2, 3],
        "BasePolicy": ["Liability", "Collision", "Liability"],
    })
    motor_df.to_csv(base_path / "motor" / "fraud_oracle.csv", index=False)
    
    # Create minimal demographics dataset
    demo_df = pd.DataFrame({
        "age": [25, 35, 45],
        "weight": [70, 80, 90],
        "height": [170, 175, 180],
        "income_lpa": [10, 20, 30],
        "smoker": ["yes", "no", "yes"],
        "city": ["Delhi", "Mumbai", "Bangalore"],
        "occupation": ["Engineer", "Doctor", "Lawyer"],
        "insurance_premium_category": ["low", "medium", "high"],
    })
    demo_df.to_csv(base_path / "demographics" / "Indian_Insurance_Data.csv", index=False)
    
    # Create minimal health Train datasets
    health_main_train = pd.DataFrame({
        "Provider": [1, 2, 3],
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012009", "02012009", "03012009"],
        "ClaimEndDt": ["31012009", "28022009", "31032009"],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
        "PotentialFraud": ["No", "Yes", "No"],
    })
    health_main_train.to_csv(base_path / "health" / "Train-1542865627584.csv", index=False)
    
    # Create minimal health Train Beneficiary dataset
    health_bene_train = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "DOB": ["01011950", "02011955", "03011960"],
        "DOD": ["", "", ""],
        "Gender": [1, 2, 1],
        "Race": [1, 2, 3],
        "RenalDiseaseIndicator": [0, 0, 1],
        "State": ["AL", "AK", "AZ"],
        "County": [1, 2, 3],
        "NoOfMonths_PartACov": [12, 12, 12],
        "NoOfMonths_PartBCov": [12, 12, 12],
        "ChronicCond_Alzheimer": [0, 0, 1],
        "ChronicCond_Heartfailure": [0, 1, 0],
        "ChronicCond_KidneyDisease": [0, 0, 1],
        "ChronicCond_Cancer": [0, 0, 0],
        "ChronicCond_ObstrPulmonary": [0, 0, 0],
        "ChronicCond_Depression": [0, 1, 0],
        "ChronicCond_Diabetes": [1, 0, 0],
        "ChronicCond_IschemicHeart": [0, 0, 0],
        "ChronicCond_Osteoporasis": [0, 0, 0],
        "ChronicCond_rheumatoidarthritis": [0, 0, 0],
        "ChronicCond_stroke": [0, 0, 0],
        "IPAnnualReimbursementAmt": [10000, 20000, 30000],
        "IPAnnualDeductibleAmt": [1000, 2000, 3000],
        "OPAnnualReimbursementAmt": [5000, 10000, 15000],
        "OPAnnualDeductibleAmt": [500, 1000, 1500],
    })
    health_bene_train.to_csv(base_path / "health" / "Train_Beneficiarydata-1542865627584.csv", index=False)
    
    # Create minimal health Train Inpatient dataset
    health_inp_train = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012009", "02012009", "03012009"],
        "ClaimEndDt": ["31012009", "28022009", "31032009"],
        "Provider": [1, 2, 3],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "AdmissionDt": ["01012009", "02012009", "03012009"],
        "DischargeDt": ["31012009", "28022009", "31032009"],
        "DiagnosisGroupCode": ["001", "002", "003"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
    })
    health_inp_train.to_csv(base_path / "health" / "Train_Inpatientdata-1542865627584.csv", index=False)
    
    # Create minimal health Train Outpatient dataset
    health_out_train = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012009", "02012009", "03012009"],
        "ClaimEndDt": ["31012009", "28022009", "31032009"],
        "Provider": [1, 2, 3],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
    })
    health_out_train.to_csv(base_path / "health" / "Train_Outpatientdata-1542865627584.csv", index=False)
    
    # Create minimal health Test datasets (same structure, without PotentialFraud in main)
    health_main_test = pd.DataFrame({
        "Provider": [1, 2, 3],
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012010", "02012010", "03012010"],
        "ClaimEndDt": ["31012010", "28022010", "31032010"],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
    })
    health_main_test.to_csv(base_path / "health" / "Test-1542969243754.csv", index=False)
    
    # Create minimal health Test Beneficiary dataset
    health_bene_test = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "DOB": ["01011950", "02011955", "03011960"],
        "DOD": ["", "", ""],
        "Gender": [1, 2, 1],
        "Race": [1, 2, 3],
        "RenalDiseaseIndicator": [0, 0, 1],
        "State": ["AL", "AK", "AZ"],
        "County": [1, 2, 3],
        "NoOfMonths_PartACov": [12, 12, 12],
        "NoOfMonths_PartBCov": [12, 12, 12],
        "ChronicCond_Alzheimer": [0, 0, 1],
        "ChronicCond_Heartfailure": [0, 1, 0],
        "ChronicCond_KidneyDisease": [0, 0, 1],
        "ChronicCond_Cancer": [0, 0, 0],
        "ChronicCond_ObstrPulmonary": [0, 0, 0],
        "ChronicCond_Depression": [0, 1, 0],
        "ChronicCond_Diabetes": [1, 0, 0],
        "ChronicCond_IschemicHeart": [0, 0, 0],
        "ChronicCond_Osteoporasis": [0, 0, 0],
        "ChronicCond_rheumatoidarthritis": [0, 0, 0],
        "ChronicCond_stroke": [0, 0, 0],
        "IPAnnualReimbursementAmt": [10000, 20000, 30000],
        "IPAnnualDeductibleAmt": [1000, 2000, 3000],
        "OPAnnualReimbursementAmt": [5000, 10000, 15000],
        "OPAnnualDeductibleAmt": [500, 1000, 1500],
    })
    health_bene_test.to_csv(base_path / "health" / "Test_Beneficiarydata-1542969243754.csv", index=False)
    
    # Create minimal health Test Inpatient dataset
    health_inp_test = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012010", "02012010", "03012010"],
        "ClaimEndDt": ["31012010", "28022010", "31032010"],
        "Provider": [1, 2, 3],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "AdmissionDt": ["01012010", "02012010", "03012010"],
        "DischargeDt": ["31012010", "28022010", "31032010"],
        "DiagnosisGroupCode": ["001", "002", "003"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
    })
    health_inp_test.to_csv(base_path / "health" / "Test_Inpatientdata-1542969243754.csv", index=False)
    
    # Create minimal health Test Outpatient dataset
    health_out_test = pd.DataFrame({
        "BeneID": [100, 101, 102],
        "ClaimID": [1001, 1002, 1003],
        "ClaimStartDt": ["01012010", "02012010", "03012010"],
        "ClaimEndDt": ["31012010", "28022010", "31032010"],
        "Provider": [1, 2, 3],
        "InscClaimAmtReimbursed": [1000, 2000, 3000],
        "AttendingPhysician": [100, 101, 102],
        "OperatingPhysician": [100, 101, 102],
        "OtherPhysician": [100, 101, 102],
        "DeductibleAmtPaid": [100, 200, 300],
        "ClmAdmitDiagnosisCode": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_1": ["78650", "78651", "78652"],
        "ClmDiagnosisCode_2": ["", "", ""],
        "ClmDiagnosisCode_3": ["", "", ""],
        "ClmDiagnosisCode_4": ["", "", ""],
        "ClmDiagnosisCode_5": ["", "", ""],
        "ClmDiagnosisCode_6": ["", "", ""],
        "ClmDiagnosisCode_7": ["", "", ""],
        "ClmDiagnosisCode_8": ["", "", ""],
        "ClmDiagnosisCode_9": ["", "", ""],
        "ClmDiagnosisCode_10": ["", "", ""],
        "ClmProcedureCode_1": ["99213", "99214", "99215"],
        "ClmProcedureCode_2": ["", "", ""],
        "ClmProcedureCode_3": ["", "", ""],
        "ClmProcedureCode_4": ["", "", ""],
        "ClmProcedureCode_5": ["", "", ""],
        "ClmProcedureCode_6": ["", "", ""],
    })
    health_out_test.to_csv(base_path / "health" / "Test_Outpatientdata-1542969243754.csv", index=False)
    
    print("✓ Test datasets created successfully")

if __name__ == "__main__":
    setup_test_datasets()