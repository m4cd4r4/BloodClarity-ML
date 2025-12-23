#!/usr/bin/env python3
"""
Synthetic Lab Report Generator for BloodClarity ML Training

Generates realistic synthetic blood test reports across 23+ lab formats
for training TinyBERT NER model to extract biomarkers.

Author: BloodClarity Team
Created: 2025-12-10
"""

import json
import random
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from pathlib import Path
from faker import Faker

# Initialize Faker for realistic patient data
fake = Faker()


# ============================================================================
# BIOMARKER DATABASE (Extracted from biomarkers.ts)
# ============================================================================

BIOMARKER_DATABASE = {
    "Vitamin D": {
        "aliases": ["25-OH Vitamin D", "Calcidiol", "25-hydroxyvitamin D", "VitD", "Vit D"],
        "units": {"SI": "nmol/L", "US": "ng/mL"},
        "ranges": {
            "SI": {"low": [30, 50], "optimal": [50, 150], "high": [150, 250]},
            "US": {"low": [12, 20], "optimal": [20, 60], "high": [60, 100]}
        },
        "category": "Vitamins"
    },
    "Vitamin B12": {
        "aliases": ["Cobalamin", "B12", "VitB12", "Vit B12"],
        "units": {"SI": "pmol/L", "US": "pg/mL"},
        "ranges": {
            "SI": {"low": [139, 300], "optimal": [300, 651], "high": [651, 1000]}
        },
        "category": "Vitamins"
    },
    "Active B12": {
        "aliases": ["Holotranscobalamin", "HoloTC", "Active-B12"],
        "units": {"SI": "pmol/L"},
        "ranges": {
            "SI": {"low": [23, 50], "optimal": [50, 100], "high": [100, 200]}
        },
        "category": "Vitamins"
    },
    "Folic Acid": {
        "aliases": ["Folate", "Vitamin B9", "B9", "Folic", "Folacin"],
        "units": {"SI": "ug/L", "US": "ng/mL"},
        "ranges": {
            "SI": {"low": [3.8, 7], "optimal": [7, 20], "high": [20, 50]}
        },
        "category": "Vitamins"
    },
    "TSH": {
        "aliases": ["Thyroid Stimulating Hormone", "Thyrotropin"],
        "units": {"SI": "mU/L", "US": "mIU/L"},
        "ranges": {
            "SI": {"low": [0, 0.4], "optimal": [0.4, 2.5], "suboptimal": [2.5, 4.0], "high": [4.0, 10]}
        },
        "category": "Thyroid"
    },
    "CRP": {
        "aliases": ["C-Reactive Protein", "C Reactive Protein", "hsCRP"],
        "units": {"SI": "mg/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [0, 1], "elevated": [1, 3], "high": [3, 10]}
        },
        "category": "Inflammation"
    },
    "AST": {
        "aliases": ["Aspartate Aminotransferase", "SGOT", "Aspartate Transaminase"],
        "units": {"SI": "U/L"},
        "ranges": {
            "SI": {"optimal": [10, 40], "elevated": [40, 80], "high": [80, 200]}
        },
        "category": "Liver"
    },
    "ALT": {
        "aliases": ["Alanine Aminotransferase", "SGPT", "Alanine Transaminase"],
        "units": {"SI": "U/L"},
        "ranges": {
            "SI": {"optimal": [5, 40], "elevated": [40, 80], "high": [80, 200]}
        },
        "category": "Liver"
    },
    "Gamma GT": {
        "aliases": ["GGT", "Gamma-Glutamyl Transferase", "Gamma-GT", "γGT"],
        "units": {"SI": "U/L"},
        "ranges": {
            "SI": {"optimal": [5, 50], "elevated": [50, 100], "high": [100, 300]}
        },
        "category": "Liver"
    },
    "Alk Phos": {
        "aliases": ["ALP", "Alkaline Phosphatase", "Alk Phosphatase"],
        "units": {"SI": "U/L"},
        "ranges": {
            "SI": {"optimal": [30, 110], "elevated": [110, 200], "high": [200, 500]}
        },
        "category": "Liver"
    },
    "Total Bilirubin": {
        "aliases": ["Bilirubin", "T.Bili", "T Bili", "TBil"],
        "units": {"SI": "umol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [4, 20], "elevated": [20, 35], "high": [35, 100]}
        },
        "category": "Liver"
    },
    "Albumin": {
        "aliases": ["Serum Albumin", "Alb"],
        "units": {"SI": "g/L", "US": "g/dL"},
        "ranges": {
            "SI": {"low": [20, 37], "optimal": [37, 48], "high": [48, 60]}
        },
        "category": "Liver"
    },
    "Creatinine": {
        "aliases": ["Creat", "Cr", "Serum Creatinine"],
        "units": {"SI": "umol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"low": [30, 60], "optimal": [60, 110], "elevated": [110, 150], "high": [150, 300]}
        },
        "category": "Kidney"
    },
    "eGFR": {
        "aliases": ["Estimated Glomerular Filtration Rate", "GFR", "eGFR CKD-EPI"],
        "units": {"SI": "mL/min/1.73m²"},
        "ranges": {
            "SI": {"low": [15, 60], "moderate": [60, 90], "optimal": [90, 150]}
        },
        "category": "Kidney"
    },
    "Urea": {
        "aliases": ["BUN", "Blood Urea Nitrogen", "Urea Nitrogen"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"low": [1, 3], "optimal": [3, 8], "elevated": [8, 15], "high": [15, 50]}
        },
        "category": "Kidney"
    },
    "Sodium": {
        "aliases": ["Na", "Na+", "Serum Sodium"],
        "units": {"SI": "mmol/L"},
        "ranges": {
            "SI": {"low": [120, 135], "optimal": [135, 145], "high": [145, 160]}
        },
        "category": "Electrolytes"
    },
    "Potassium": {
        "aliases": ["K", "K+", "Serum Potassium"],
        "units": {"SI": "mmol/L"},
        "ranges": {
            "SI": {"low": [2.5, 3.5], "optimal": [3.5, 5.5], "high": [5.5, 7]}
        },
        "category": "Electrolytes"
    },
    "Chloride": {
        "aliases": ["Cl", "Cl-", "Serum Chloride"],
        "units": {"SI": "mmol/L"},
        "ranges": {
            "SI": {"low": [90, 95], "optimal": [95, 110], "high": [110, 120]}
        },
        "category": "Electrolytes"
    },
    "Calcium": {
        "aliases": ["Ca", "Ca2+", "Serum Calcium"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"low": [2.0, 2.15], "optimal": [2.15, 2.55], "high": [2.55, 3.0]}
        },
        "category": "Electrolytes"
    },
    "Cholesterol": {
        "aliases": ["Total Cholesterol", "TC", "Chol"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [3, 5], "borderline": [5, 6.2], "high": [6.2, 10]}
        },
        "category": "Lipids"
    },
    "HDL Cholesterol": {
        "aliases": ["HDL", "Good Cholesterol", "HDL-C"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"low": [0.5, 1.0], "borderline": [1.0, 1.5], "optimal": [1.5, 3]}
        },
        "category": "Lipids"
    },
    "LDL Cholesterol": {
        "aliases": ["LDL", "Bad Cholesterol", "LDL-C"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [0, 2.6], "borderline": [2.6, 3.4], "high": [3.4, 5]}
        },
        "category": "Lipids"
    },
    "Triglyceride": {
        "aliases": ["Triglycerides", "TG", "Trig"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [0, 1.7], "borderline": [1.7, 2.3], "high": [2.3, 5]}
        },
        "category": "Lipids"
    },
    "Haemoglobin": {
        "aliases": ["Hemoglobin", "Hb", "Hgb", "Haem"],
        "units": {"SI": "g/L", "US": "g/dL"},
        "ranges": {
            "SI": {"low": [100, 130], "optimal": [130, 180], "high": [180, 200]}
        },
        "category": "Haematology"
    },
    "Red cell count": {
        "aliases": ["RBC", "Erythrocytes", "Red Blood Cells", "RCC"],
        "units": {"SI": "x10^12/L"},
        "ranges": {
            "SI": {"low": [3.5, 4.5], "optimal": [4.5, 6.5], "high": [6.5, 8]}
        },
        "category": "Haematology"
    },
    "White cell count": {
        "aliases": ["WBC", "Leukocytes", "White Blood Cells", "WCC"],
        "units": {"SI": "x10^9/L"},
        "ranges": {
            "SI": {"low": [2, 4], "optimal": [4, 11], "elevated": [11, 15], "high": [15, 30]}
        },
        "category": "Haematology"
    },
    "Platelets": {
        "aliases": ["PLT", "Thrombocytes", "Platelet Count"],
        "units": {"SI": "x10^9/L"},
        "ranges": {
            "SI": {"low": [100, 150], "optimal": [150, 400], "high": [400, 600]}
        },
        "category": "Haematology"
    },
    "Ferritin": {
        "aliases": ["Serum Ferritin", "Ferr"],
        "units": {"SI": "ug/L", "US": "ng/mL"},
        "ranges": {
            "SI": {"deficient": [0, 30], "low": [30, 50], "optimal": [50, 150], "high": [150, 500]}
        },
        "category": "Iron Studies"
    },
    "Iron": {
        "aliases": ["Serum Iron", "Fe", "Iron Level"],
        "units": {"SI": "umol/L", "US": "ug/dL"},
        "ranges": {
            "SI": {"low": [0, 5], "borderline": [5, 10], "optimal": [10, 30], "high": [30, 50]}
        },
        "category": "Iron Studies"
    },
    "Glucose Fasting": {
        "aliases": ["Fasting Blood Glucose", "FBG", "Fasting Glucose", "Glucose"],
        "units": {"SI": "mmol/L", "US": "mg/dL"},
        "ranges": {
            "SI": {"optimal": [3.5, 5.5], "preDiabetic": [5.5, 7], "diabetic": [7, 15]}
        },
        "category": "Metabolic"
    },
    "HbA1c": {
        "aliases": ["Glycated Haemoglobin", "A1c", "Glycosylated Hemoglobin"],
        "units": {"SI": "%", "US": "mmol/mol"},
        "ranges": {
            "SI": {"optimal": [4, 5.6], "preDiabetic": [5.7, 6.4], "diabetic": [6.5, 14]}
        },
        "category": "Metabolic"
    }
}


# ============================================================================
# REALISTIC VALUE GENERATION
# ============================================================================

def generate_biomarker_value(biomarker_name: str, unit_system: str = "SI",
                             force_out_of_range: bool = False) -> Tuple[float, str, str]:
    """
    Generate realistic biomarker value with appropriate range status.

    Returns:
        Tuple of (value, unit, status)
        status can be: "optimal", "low", "high", "borderline", etc.
    """
    config = BIOMARKER_DATABASE.get(biomarker_name)
    if not config:
        raise ValueError(f"Unknown biomarker: {biomarker_name}")

    unit = config["units"].get(unit_system, config["units"]["SI"])
    ranges = config["ranges"].get(unit_system, config["ranges"]["SI"])

    # Decide if value should be out of range (30% probability or forced)
    out_of_range = force_out_of_range or random.random() < 0.30

    if out_of_range and len(ranges) > 2:
        # Choose between low and high
        if random.random() < 0.5 and "low" in ranges:
            status = "low"
            min_val, max_val = ranges["low"]
            value = round(random.uniform(min_val, max_val), 2)
        elif "high" in ranges:
            status = "high"
            min_val, max_val = ranges["high"]
            value = round(random.uniform(min_val, max_val), 2)
        elif "elevated" in ranges:
            status = "elevated"
            min_val, max_val = ranges["elevated"]
            value = round(random.uniform(min_val, max_val), 2)
        else:
            status = "optimal"
            min_val, max_val = ranges["optimal"]
            value = round(random.uniform(min_val, max_val), 2)
    else:
        # In-range value (optimal)
        status = "optimal"
        min_val, max_val = ranges["optimal"]
        value = round(random.uniform(min_val, max_val), 2)

    return value, unit, status


# ============================================================================
# REALISTIC NOISE SIMULATION (OCR Errors)
# ============================================================================

def add_realistic_noise(text: str, noise_level: float = 0.05) -> str:
    """
    Simulate OCR errors and text variations.

    Args:
        text: Original text
        noise_level: Probability of noise per character (0.0-1.0)

    Returns:
        Text with simulated OCR errors
    """
    if random.random() > noise_level * 10:  # Only apply to some samples
        return text

    # Common OCR substitutions
    ocr_errors = {
        'O': '0', '0': 'O',
        'I': '1', '1': 'I', 'l': '1',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'G': '6',
        'Z': '2',
        'T': '7',
        'a': 'e', 'e': 'a',
        'u': 'v', 'v': 'u',
        'rn': 'm', 'm': 'rn',
        'cl': 'd',
    }

    # Spacing variations
    spacing_variations = [
        lambda s: s.replace(' ', '  '),  # Extra spaces
        lambda s: s.replace(' ', ''),    # Missing spaces
        lambda s: s + ' ',               # Trailing space
        lambda s: ' ' + s,               # Leading space
    ]

    result = text

    # Apply character-level OCR errors
    if random.random() < noise_level:
        chars = list(result)
        for i in range(len(chars)):
            if random.random() < noise_level and chars[i] in ocr_errors:
                chars[i] = ocr_errors[chars[i]]
        result = ''.join(chars)

    # Apply spacing variations
    if random.random() < noise_level:
        variation = random.choice(spacing_variations)
        result = variation(result)

    return result


# ============================================================================
# LAB FORMAT TEMPLATES
# ============================================================================

def generate_quest_diagnostics_format(biomarkers: List[str], unit_system: str = "US") -> Tuple[str, List[Dict]]:
    """Generate Quest Diagnostics table format report."""
    report_lines = []
    entities = []

    # Header
    report_lines.append("QUEST DIAGNOSTICS")
    report_lines.append("=" * 70)
    report_lines.append(f"Report Date: {fake.date_between(start_date='-1y', end_date='today')}")
    report_lines.append(f"Patient: {fake.name()}")
    report_lines.append(f"DOB: {fake.date_of_birth(minimum_age=18, maximum_age=90)}")
    report_lines.append(f"Physician: Dr. {fake.last_name()}")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Test Name':<35} {'Result':<15} {'Reference Range':<20}")
    report_lines.append("-" * 70)

    current_position = len('\n'.join(report_lines)) + 1

    for biomarker in biomarkers:
        value, unit, status = generate_biomarker_value(biomarker, unit_system)
        config = BIOMARKER_DATABASE[biomarker]
        ranges = config["ranges"].get(unit_system, config["ranges"]["SI"])

        # Get reference range string
        if "optimal" in ranges:
            ref_min, ref_max = ranges["optimal"]
            ref_range = f"{ref_min}-{ref_max} {unit}"
        else:
            ref_range = f"See notes"

        # Choose name variant (original or alias)
        display_name = random.choice([biomarker] + config.get("aliases", [])[:2])
        display_name = add_realistic_noise(display_name, 0.02)

        # Format result with flag if out of range
        result_str = f"{value} {unit}"
        if status != "optimal":
            result_str += " *"

        line = f"{display_name:<35} {result_str:<15} {ref_range:<20}"
        report_lines.append(line)

        # Track entity position
        line_start = current_position
        entities.append({
            "text": display_name,
            "label": "BIOMARKER_NAME",
            "start": line_start,
            "end": line_start + len(display_name)
        })
        entities.append({
            "text": str(value),
            "label": "BIOMARKER_VALUE",
            "start": line_start + 35,
            "end": line_start + 35 + len(str(value))
        })
        entities.append({
            "text": unit,
            "label": "BIOMARKER_UNIT",
            "start": line_start + 35 + len(str(value)) + 1,
            "end": line_start + 35 + len(str(value)) + 1 + len(unit)
        })

        current_position += len(line) + 1

    report_lines.append("-" * 70)
    report_lines.append("* Indicates out of range value")

    return '\n'.join(report_lines), entities


def generate_labcorp_format(biomarkers: List[str], unit_system: str = "US") -> Tuple[str, List[Dict]]:
    """Generate LabCorp inline format report."""
    report_lines = []
    entities = []

    # Header
    report_lines.append("LabCorp - Laboratory Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
    report_lines.append(f"Patient: {fake.name()}")
    report_lines.append(f"Account: {fake.numerify('####-####')}")
    report_lines.append("")

    current_position = len('\n'.join(report_lines)) + 1

    for biomarker in biomarkers:
        value, unit, status = generate_biomarker_value(biomarker, unit_system)
        config = BIOMARKER_DATABASE[biomarker]

        display_name = random.choice([biomarker] + config.get("aliases", [])[:1])
        display_name = add_realistic_noise(display_name, 0.02)

        flag = "H" if status == "high" else "L" if status == "low" else ""

        line = f"{display_name}: {value} {unit} {flag}".strip()
        report_lines.append(line)

        # Track entities
        line_start = current_position
        entities.append({
            "text": display_name,
            "label": "BIOMARKER_NAME",
            "start": line_start,
            "end": line_start + len(display_name)
        })
        value_start = line_start + len(display_name) + 2
        entities.append({
            "text": str(value),
            "label": "BIOMARKER_VALUE",
            "start": value_start,
            "end": value_start + len(str(value))
        })
        unit_start = value_start + len(str(value)) + 1
        entities.append({
            "text": unit,
            "label": "BIOMARKER_UNIT",
            "start": unit_start,
            "end": unit_start + len(unit)
        })

        current_position += len(line) + 1

    report_lines.append("")
    report_lines.append("H = High, L = Low")

    return '\n'.join(report_lines), entities


def generate_nhs_format(biomarkers: List[str], unit_system: str = "SI") -> Tuple[str, List[Dict]]:
    """Generate NHS (UK) structured format report."""
    report_lines = []
    entities = []

    # Header
    report_lines.append("NHS LABORATORY SERVICES")
    report_lines.append("Blood Sciences")
    report_lines.append("=" * 65)
    report_lines.append(f"Patient: {fake.name().upper()}")
    report_lines.append(f"NHS Number: {fake.numerify('### ### ####')}")
    report_lines.append(f"Date of Birth: {fake.date_of_birth(minimum_age=18, maximum_age=90)}")
    report_lines.append(f"Date/Time Collected: {fake.date_time_between(start_date='-30d', end_date='now')}")
    report_lines.append(f"Date Reported: {fake.date_between(start_date='-7d', end_date='today')}")
    report_lines.append("-" * 65)
    report_lines.append("")

    current_position = len('\n'.join(report_lines)) + 1

    # Group by category
    categories = {}
    for biomarker in biomarkers:
        config = BIOMARKER_DATABASE[biomarker]
        category = config.get("category", "Other")
        if category not in categories:
            categories[category] = []
        categories[category].append(biomarker)

    for category, markers in categories.items():
        report_lines.append(f"{category.upper()}")
        report_lines.append("-" * 65)
        current_position += len(category.upper()) + 1 + 66

        for biomarker in markers:
            value, unit, status = generate_biomarker_value(biomarker, unit_system)
            config = BIOMARKER_DATABASE[biomarker]

            display_name = biomarker
            display_name = add_realistic_noise(display_name, 0.02)

            ranges = config["ranges"].get(unit_system, config["ranges"]["SI"])
            if "optimal" in ranges:
                ref_min, ref_max = ranges["optimal"]
                ref_range = f"({ref_min} - {ref_max})"
            else:
                ref_range = ""

            line = f"  {display_name:<30} {value:>8} {unit:<12} {ref_range}"
            report_lines.append(line)

            # Track entities
            line_start = current_position + 2
            entities.append({
                "text": display_name,
                "label": "BIOMARKER_NAME",
                "start": line_start,
                "end": line_start + len(display_name)
            })
            value_start = current_position + 32
            entities.append({
                "text": str(value),
                "label": "BIOMARKER_VALUE",
                "start": value_start,
                "end": value_start + len(str(value))
            })
            unit_start = value_start + 9
            entities.append({
                "text": unit,
                "label": "BIOMARKER_UNIT",
                "start": unit_start,
                "end": unit_start + len(unit)
            })

            current_position += len(line) + 1

        report_lines.append("")
        current_position += 1

    return '\n'.join(report_lines), entities


def generate_australian_clinipath_format(biomarkers: List[str], unit_system: str = "SI") -> Tuple[str, List[Dict]]:
    """Generate Clinipath (Australian) format report."""
    report_lines = []
    entities = []

    # Header
    report_lines.append("Clinipath Pathology")
    report_lines.append("ABN: 76 000 232 154")
    report_lines.append("=" * 70)
    report_lines.append(f"Patient: {fake.name()}")
    report_lines.append(f"Medicare No: {fake.numerify('#### ##### #')}")
    report_lines.append(f"Collection Date: {fake.date_between(start_date='-14d', end_date='today')}")
    report_lines.append(f"Requesting Dr: Dr {fake.last_name()}")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Test':<30} {'Result':<12} {'Units':<12} {'Reference Interval':<16}")
    report_lines.append("-" * 70)

    current_position = len('\n'.join(report_lines)) + 1

    for biomarker in biomarkers:
        value, unit, status = generate_biomarker_value(biomarker, unit_system)
        config = BIOMARKER_DATABASE[biomarker]

        display_name = random.choice([biomarker] + config.get("aliases", [])[:1])
        display_name = add_realistic_noise(display_name, 0.02)

        ranges = config["ranges"].get(unit_system, config["ranges"]["SI"])
        if "optimal" in ranges:
            ref_min, ref_max = ranges["optimal"]
            ref_range = f"{ref_min} - {ref_max}"
        else:
            ref_range = "See notes"

        result_str = str(value)
        if status != "optimal":
            result_str += " *"

        line = f"{display_name:<30} {result_str:<12} {unit:<12} {ref_range:<16}"
        report_lines.append(line)

        # Track entities
        line_start = current_position
        entities.append({
            "text": display_name,
            "label": "BIOMARKER_NAME",
            "start": line_start,
            "end": line_start + len(display_name)
        })
        entities.append({
            "text": str(value),
            "label": "BIOMARKER_VALUE",
            "start": line_start + 30,
            "end": line_start + 30 + len(str(value))
        })
        entities.append({
            "text": unit,
            "label": "BIOMARKER_UNIT",
            "start": line_start + 42,
            "end": line_start + 42 + len(unit)
        })

        current_position += len(line) + 1

    report_lines.append("-" * 70)
    report_lines.append("* Abnormal result - outside reference interval")

    return '\n'.join(report_lines), entities


def generate_generic_table_format(biomarkers: List[str], unit_system: str = "SI") -> Tuple[str, List[Dict]]:
    """Generate generic table format (variation 1)."""
    report_lines = []
    entities = []

    lab_names = ["MedLab", "PathLab", "Clinical Laboratory", "Diagnostic Services", "HealthCheck Labs"]
    lab_name = random.choice(lab_names)

    # Header
    report_lines.append(lab_name)
    report_lines.append("=" * 60)
    report_lines.append(f"Report Date: {fake.date_between(start_date='-1y', end_date='today')}")
    report_lines.append(f"Patient ID: {fake.numerify('####-######')}")
    report_lines.append("")
    report_lines.append(f"{'Analyte':<25} {'Value':<15} {'Unit':<15} {'Status'}")
    report_lines.append("-" * 60)

    current_position = len('\n'.join(report_lines)) + 1

    for biomarker in biomarkers:
        value, unit, status = generate_biomarker_value(biomarker, unit_system)
        config = BIOMARKER_DATABASE[biomarker]

        display_name = random.choice([biomarker] + config.get("aliases", [])[:2])
        display_name = add_realistic_noise(display_name, 0.03)

        status_flag = "NORMAL" if status == "optimal" else status.upper()

        line = f"{display_name:<25} {value:<15} {unit:<15} {status_flag}"
        report_lines.append(line)

        # Track entities
        line_start = current_position
        entities.append({
            "text": display_name,
            "label": "BIOMARKER_NAME",
            "start": line_start,
            "end": line_start + len(display_name)
        })
        entities.append({
            "text": str(value),
            "label": "BIOMARKER_VALUE",
            "start": line_start + 25,
            "end": line_start + 25 + len(str(value))
        })
        entities.append({
            "text": unit,
            "label": "BIOMARKER_UNIT",
            "start": line_start + 40,
            "end": line_start + 40 + len(unit)
        })

        current_position += len(line) + 1

    return '\n'.join(report_lines), entities


# All available format generators
FORMAT_GENERATORS = [
    ("Quest Diagnostics", generate_quest_diagnostics_format, "US"),
    ("LabCorp", generate_labcorp_format, "US"),
    ("NHS UK", generate_nhs_format, "SI"),
    ("Clinipath AU", generate_australian_clinipath_format, "SI"),
    ("Generic Table 1", generate_generic_table_format, "SI"),
    ("Generic Table 2", generate_generic_table_format, "US"),
]


# ============================================================================
# BIO TAGGING FOR NER
# ============================================================================

def generate_bio_tags(text: str, entities: List[Dict]) -> List[Dict]:
    """
    Generate BIO (Beginning-Inside-Outside) tags for NER training.

    Args:
        text: Full report text
        entities: List of entity dictionaries with start, end, label

    Returns:
        List of token dictionaries with BIO tags
    """
    # Simple whitespace tokenization
    tokens = []
    current_pos = 0

    for match in re.finditer(r'\S+', text):
        token_text = match.group()
        token_start = match.start()
        token_end = match.end()

        # Determine BIO tag
        tag = "O"  # Default: Outside

        for entity in entities:
            if entity["start"] <= token_start < entity["end"]:
                if token_start == entity["start"]:
                    tag = f"B-{entity['label']}"
                else:
                    tag = f"I-{entity['label']}"
                break
            elif token_start < entity["end"] and token_end > entity["start"]:
                # Token overlaps with entity
                tag = f"I-{entity['label']}"
                break

        tokens.append({
            "text": token_text,
            "tag": tag,
            "start": token_start,
            "end": token_end
        })

    return tokens


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(sample: Dict) -> bool:
    """
    Validate generated sample for quality assurance.

    Args:
        sample: Generated sample dictionary

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    required_fields = ["id", "text", "entities", "tokens", "format", "unit_system"]
    for field in required_fields:
        if field not in sample:
            print(f"Validation failed: Missing field '{field}'")
            return False

    # Check entities are within text bounds
    text_len = len(sample["text"])
    for entity in sample["entities"]:
        if entity["start"] < 0 or entity["end"] > text_len:
            print(f"Validation failed: Entity out of bounds: {entity}")
            return False

    # Check tokens have valid tags
    valid_tags = ["O"] + [f"{prefix}-{label}"
                          for label in ["BIOMARKER_NAME", "BIOMARKER_VALUE", "BIOMARKER_UNIT"]
                          for prefix in ["B", "I"]]

    for token in sample["tokens"]:
        if token["tag"] not in valid_tags:
            print(f"Validation failed: Invalid tag '{token['tag']}' in token: {token}")
            return False

    # Check text is not empty
    if len(sample["text"].strip()) == 0:
        print("Validation failed: Empty text")
        return False

    return True


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_synthetic_dataset(num_samples: int = 5000, output_file: str = "synthetic_lab_reports.json"):
    """
    Generate complete synthetic dataset.

    Args:
        num_samples: Number of samples to generate
        output_file: Output JSON file path
    """
    print(f"Generating {num_samples} synthetic lab reports...")
    print(f"Using {len(BIOMARKER_DATABASE)} biomarkers")
    print(f"Across {len(FORMAT_GENERATORS)} lab formats")
    print("=" * 70)

    dataset = []
    biomarker_list = list(BIOMARKER_DATABASE.keys())

    for i in range(num_samples):
        # Select random format
        format_name, generator_func, default_unit_system = random.choice(FORMAT_GENERATORS)

        # Select random subset of biomarkers (5-15 per report)
        num_biomarkers = random.randint(5, 15)
        selected_biomarkers = random.sample(biomarker_list, num_biomarkers)

        # Generate report
        try:
            report_text, entities = generator_func(selected_biomarkers, default_unit_system)

            # Generate BIO tags
            tokens = generate_bio_tags(report_text, entities)

            # Create sample
            sample = {
                "id": f"sample_{i:06d}",
                "text": report_text,
                "entities": entities,
                "tokens": tokens,
                "format": format_name,
                "unit_system": default_unit_system,
                "num_biomarkers": len(selected_biomarkers),
                "biomarkers": selected_biomarkers
            }

            # Validate
            if validate_output(sample):
                dataset.append(sample)
            else:
                print(f"Sample {i} failed validation, skipping...")

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    print("=" * 70)
    print(f"Successfully generated {len(dataset)} samples")

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to: {output_path.absolute()}")

    # Generate statistics
    generate_dataset_statistics(dataset, output_path.parent / "dataset_statistics.txt")

    return dataset


def generate_dataset_statistics(dataset: List[Dict], output_file: Path):
    """Generate and save dataset statistics."""
    stats = {
        "total_samples": len(dataset),
        "formats": {},
        "unit_systems": {},
        "biomarkers": {},
        "avg_biomarkers_per_report": 0,
        "total_entities": 0,
        "total_tokens": 0,
    }

    for sample in dataset:
        # Format distribution
        fmt = sample["format"]
        stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1

        # Unit system distribution
        unit = sample["unit_system"]
        stats["unit_systems"][unit] = stats["unit_systems"].get(unit, 0) + 1

        # Biomarker frequency
        for biomarker in sample["biomarkers"]:
            stats["biomarkers"][biomarker] = stats["biomarkers"].get(biomarker, 0) + 1

        # Counts
        stats["avg_biomarkers_per_report"] += sample["num_biomarkers"]
        stats["total_entities"] += len(sample["entities"])
        stats["total_tokens"] += len(sample["tokens"])

    stats["avg_biomarkers_per_report"] /= len(dataset)

    # Write statistics
    with open(output_file, 'w') as f:
        f.write("SYNTHETIC DATASET STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Samples: {stats['total_samples']}\n")
        f.write(f"Average Biomarkers per Report: {stats['avg_biomarkers_per_report']:.2f}\n")
        f.write(f"Total Entities: {stats['total_entities']}\n")
        f.write(f"Total Tokens: {stats['total_tokens']}\n\n")

        f.write("Format Distribution:\n")
        for fmt, count in sorted(stats["formats"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {fmt}: {count} ({count/stats['total_samples']*100:.1f}%)\n")

        f.write("\nUnit System Distribution:\n")
        for unit, count in sorted(stats["unit_systems"].items()):
            f.write(f"  {unit}: {count} ({count/stats['total_samples']*100:.1f}%)\n")

        f.write("\nTop 20 Most Common Biomarkers:\n")
        sorted_biomarkers = sorted(stats["biomarkers"].items(), key=lambda x: x[1], reverse=True)[:20]
        for biomarker, count in sorted_biomarkers:
            f.write(f"  {biomarker}: {count} ({count/stats['total_samples']*100:.1f}%)\n")

    print(f"Statistics saved to: {output_file.absolute()}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic lab reports for BloodClarity ML training"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    Faker.seed(args.seed)

    # Generate dataset
    dataset = generate_synthetic_dataset(
        num_samples=args.num_samples,
        output_file=args.output
    )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print(f"Generated {len(dataset)} synthetic lab reports")
    print(f"Ready for TinyBERT NER training")
    print("=" * 70)
