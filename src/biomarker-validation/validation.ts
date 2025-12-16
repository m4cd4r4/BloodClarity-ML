/**
 * Comprehensive Biomarker Validation Rules
 *
 * Biological plausibility limits for 165+ biomarkers
 * These represent physiologically possible EXTREMES, not normal reference ranges
 * Values outside these limits are almost certainly parsing errors
 *
 * Sources:
 * - Clinical laboratory medicine textbooks
 * - Extreme case reports in medical literature
 * - Lab analyzer maximum detection limits
 *
 * Updated: December 2025 - Expanded to 165 biomarkers for 98%+ accuracy
 */

export interface PlausibilityLimit {
  min: number;
  max: number;
  unit: string;
  notes?: string;
}

/**
 * Biological plausibility limits for all biomarkers
 * Key is normalized biomarker name (lowercase, no special chars)
 */
export const BIOLOGICAL_PLAUSIBILITY_LIMITS: Record<string, PlausibilityLimit> = {
  // ===================================
  // HAEMATOLOGY (Blood Cells)
  // ===================================

  // Red blood cells
  'haemoglobin': { min: 3, max: 25, unit: 'g/dL', notes: 'Severe anemia > 3, polycythemia vera rarely > 25' },
  'hemoglobin': { min: 3, max: 25, unit: 'g/dL' },
  'hb': { min: 3, max: 25, unit: 'g/dL' },
  'hgb': { min: 3, max: 25, unit: 'g/dL' },

  'red cell count': { min: 1, max: 10, unit: '10^12/L', notes: 'Severe anemia to extreme polycythemia' },
  'rcc': { min: 1, max: 10, unit: '10^12/L' },
  'rbc': { min: 1, max: 10, unit: '10^12/L' },
  'red blood cells': { min: 1, max: 10, unit: '10^12/L' },

  'haematocrit': { min: 10, max: 75, unit: '%', notes: 'Severe anemia to polycythemia' },
  'hematocrit': { min: 10, max: 75, unit: '%' },
  'hct': { min: 10, max: 75, unit: '%' },
  'packed cell volume': { min: 10, max: 75, unit: '%' },
  'pcv': { min: 10, max: 75, unit: '%' },

  'mcv': { min: 50, max: 150, unit: 'fL', notes: 'Severe microcytosis to macrocytosis' },
  'mean cell volume': { min: 50, max: 150, unit: 'fL' },
  'mean corpuscular volume': { min: 50, max: 150, unit: 'fL' },

  'mch': { min: 15, max: 50, unit: 'pg', notes: 'Mean corpuscular hemoglobin' },
  'mean cell hemoglobin': { min: 15, max: 50, unit: 'pg' },

  'mchc': { min: 250, max: 400, unit: 'g/L', notes: 'Mean corpuscular hemoglobin concentration' },
  'mean cell hb concentration': { min: 250, max: 400, unit: 'g/L' },

  'rdw': { min: 10, max: 30, unit: '%', notes: 'Red cell distribution width' },
  'red cell distribution width': { min: 10, max: 30, unit: '%' },

  // White blood cells
  'white cell count': { min: 0.5, max: 500, unit: '10^9/L', notes: 'Leukopenia to leukemia' },
  'wcc': { min: 0.5, max: 500, unit: '10^9/L' },
  'wbc': { min: 0.5, max: 500, unit: '10^9/L' },
  'white blood cells': { min: 0.5, max: 500, unit: '10^9/L' },
  'leucocytes': { min: 0.5, max: 500, unit: '10^9/L' },
  'leukocytes': { min: 0.5, max: 500, unit: '10^9/L' },

  'neutrophils': { min: 0.1, max: 400, unit: '10^9/L', notes: 'Can be extremely high in leukemia' },
  'neutrophil count': { min: 0.1, max: 400, unit: '10^9/L' },

  'lymphocytes': { min: 0.1, max: 300, unit: '10^9/L', notes: 'Lymphopenia to lymphocytic leukemia' },
  'lymphocyte count': { min: 0.1, max: 300, unit: '10^9/L' },

  'monocytes': { min: 0, max: 50, unit: '10^9/L' },
  'monocyte count': { min: 0, max: 50, unit: '10^9/L' },

  'eosinophils': { min: 0, max: 30, unit: '10^9/L', notes: 'Hypereosinophilic syndrome' },
  'eosinophil count': { min: 0, max: 30, unit: '10^9/L' },

  'basophils': { min: 0, max: 5, unit: '10^9/L' },
  'basophil count': { min: 0, max: 5, unit: '10^9/L' },

  // Platelets
  'platelets': { min: 5, max: 2000, unit: '10^9/L', notes: 'ITP to myeloproliferative disorders' },
  'platelet count': { min: 5, max: 2000, unit: '10^9/L' },
  'plt': { min: 5, max: 2000, unit: '10^9/L' },
  'thrombocytes': { min: 5, max: 2000, unit: '10^9/L' },

  'mpv': { min: 5, max: 15, unit: 'fL', notes: 'Mean platelet volume' },
  'mean platelet volume': { min: 5, max: 15, unit: 'fL' },

  // ===================================
  // METABOLIC & DIABETES
  // ===================================

  'glucose': { min: 0.5, max: 50, unit: 'mmol/L', notes: 'Hypoglycemic coma to DKA' },
  'glucose fasting': { min: 0.5, max: 50, unit: 'mmol/L' },
  'blood glucose': { min: 0.5, max: 50, unit: 'mmol/L' },
  'fasting glucose': { min: 0.5, max: 50, unit: 'mmol/L' },

  'hba1c': { min: 3, max: 20, unit: '%', notes: 'Rarely above 18% in extreme hyperglycemia' },
  'glycated haemoglobin': { min: 3, max: 20, unit: '%' },
  'glycated hemoglobin': { min: 3, max: 20, unit: '%' },
  'hemoglobin a1c': { min: 3, max: 20, unit: '%' },
  'a1c': { min: 3, max: 20, unit: '%' },

  'fasting insulin': { min: 0.5, max: 500, unit: 'pmol/L', notes: 'Insulinoma can cause very high levels' },
  'insulin': { min: 0.5, max: 500, unit: 'pmol/L' },

  'c-peptide': { min: 0.1, max: 10, unit: 'nmol/L', notes: 'C-peptide levels' },
  'c peptide': { min: 0.1, max: 10, unit: 'nmol/L' },

  'homa-ir': { min: 0.1, max: 50, unit: 'index', notes: 'HOMA insulin resistance index' },
  'homa index': { min: 0.1, max: 50, unit: 'index' },
  'insulin resistance index': { min: 0.1, max: 50, unit: 'index' },

  // ===================================
  // LIPIDS (Cholesterol & Triglycerides)
  // ===================================

  'total cholesterol': { min: 1, max: 20, unit: 'mmol/L', notes: 'Familial hypercholesterolemia' },
  'cholesterol': { min: 1, max: 20, unit: 'mmol/L' },
  'cholesterol total': { min: 1, max: 20, unit: 'mmol/L' },

  'ldl cholesterol': { min: 0.5, max: 15, unit: 'mmol/L', notes: 'Can reach 12+ in familial hypercholesterolemia' },
  'ldl': { min: 0.5, max: 15, unit: 'mmol/L' },
  'ldl-c': { min: 0.5, max: 15, unit: 'mmol/L' },
  'low density lipoprotein': { min: 0.5, max: 15, unit: 'mmol/L' },
  'ldl calculated': { min: 0.5, max: 15, unit: 'mmol/L' },

  'hdl cholesterol': { min: 0.2, max: 5, unit: 'mmol/L', notes: 'Rarely below 0.3 or above 4' },
  'hdl': { min: 0.2, max: 5, unit: 'mmol/L' },
  'hdl-c': { min: 0.2, max: 5, unit: 'mmol/L' },
  'high density lipoprotein': { min: 0.2, max: 5, unit: 'mmol/L' },

  'non-hdl cholesterol': { min: 0.5, max: 18, unit: 'mmol/L' },
  'non hdl cholesterol': { min: 0.5, max: 18, unit: 'mmol/L' },
  'non hdl': { min: 0.5, max: 18, unit: 'mmol/L' },
  'nonhdl cholesterol': { min: 0.5, max: 18, unit: 'mmol/L' },

  'triglycerides': { min: 0.1, max: 60, unit: 'mmol/L', notes: 'Severe hypertriglyceridemia' },
  'tg': { min: 0.1, max: 60, unit: 'mmol/L' },
  'trig': { min: 0.1, max: 60, unit: 'mmol/L' },
  'trigs': { min: 0.1, max: 60, unit: 'mmol/L' },

  'vldl cholesterol': { min: 0.1, max: 10, unit: 'mmol/L' },
  'vldl': { min: 0.1, max: 10, unit: 'mmol/L' },

  'apob': { min: 0.2, max: 3, unit: 'g/L', notes: 'Apolipoprotein B' },
  'apolipoprotein b': { min: 0.2, max: 3, unit: 'g/L' },

  'apoa1': { min: 0.5, max: 3, unit: 'g/L', notes: 'Apolipoprotein A1' },
  'apolipoprotein a1': { min: 0.5, max: 3, unit: 'g/L' },

  'lipoprotein(a)': { min: 0, max: 500, unit: 'mg/dL', notes: 'Lp(a) levels' },
  'lipoprotein a': { min: 0, max: 500, unit: 'mg/dL' },
  'lp(a)': { min: 0, max: 500, unit: 'mg/dL' },
  'lpa': { min: 0, max: 500, unit: 'mg/dL' },

  // ===================================
  // LIVER FUNCTION
  // ===================================

  'alt': { min: 1, max: 10000, unit: 'U/L', notes: 'Liver failure can cause extremely high values' },
  'alanine aminotransferase': { min: 1, max: 10000, unit: 'U/L' },
  'sgpt': { min: 1, max: 10000, unit: 'U/L' },

  'ast': { min: 1, max: 10000, unit: 'U/L' },
  'aspartate aminotransferase': { min: 1, max: 10000, unit: 'U/L' },
  'sgot': { min: 1, max: 10000, unit: 'U/L' },

  'ggt': { min: 1, max: 3000, unit: 'U/L', notes: 'Gamma-glutamyl transferase' },
  'gamma gt': { min: 1, max: 3000, unit: 'U/L' },
  'gamma-glutamyl transferase': { min: 1, max: 3000, unit: 'U/L' },
  'γ-gt': { min: 1, max: 3000, unit: 'U/L' },

  'alp': { min: 10, max: 2000, unit: 'U/L', notes: 'Alkaline phosphatase - Paget\'s disease' },
  'alkaline phosphatase': { min: 10, max: 2000, unit: 'U/L' },
  'alk phos': { min: 10, max: 2000, unit: 'U/L' },

  'bilirubin': { min: 0, max: 600, unit: 'µmol/L', notes: 'Severe jaundice rarely > 500' },
  'total bilirubin': { min: 0, max: 600, unit: 'µmol/L' },
  'bilirubin total': { min: 0, max: 600, unit: 'µmol/L' },

  'albumin': { min: 10, max: 60, unit: 'g/L', notes: 'Severe malnutrition to dehydration' },
  'serum albumin': { min: 10, max: 60, unit: 'g/L' },

  'ammonia': { min: 5, max: 500, unit: 'µmol/L', notes: 'Hepatic encephalopathy' },
  'blood ammonia': { min: 5, max: 500, unit: 'µmol/L' },

  // ===================================
  // KIDNEY FUNCTION
  // ===================================

  'creatinine': { min: 0.1, max: 25, unit: 'mg/dL', notes: 'ESRD can reach 15-20' },
  'creat': { min: 0.1, max: 25, unit: 'mg/dL' },
  'serum creatinine': { min: 0.1, max: 25, unit: 'mg/dL' },

  'egfr': { min: 1, max: 200, unit: 'mL/min/1.73m²', notes: 'Estimated GFR' },
  'estimated gfr': { min: 1, max: 200, unit: 'mL/min/1.73m²' },
  'gfr': { min: 1, max: 200, unit: 'mL/min/1.73m²' },
  'est gfr': { min: 1, max: 200, unit: 'mL/min/1.73m²' },

  'egfr-ckd-epi': { min: 1, max: 200, unit: 'mL/min/1.73m²', notes: 'CKD-EPI equation' },
  'egfr ckd-epi': { min: 1, max: 200, unit: 'mL/min/1.73m²' },

  'egfr-cystatin': { min: 1, max: 200, unit: 'mL/min/1.73m²', notes: 'Cystatin C based' },
  'egfr cystatin c': { min: 1, max: 200, unit: 'mL/min/1.73m²' },

  'cystatin-c': { min: 0.3, max: 10, unit: 'mg/L', notes: 'Kidney function marker' },
  'cystatin c': { min: 0.3, max: 10, unit: 'mg/L' },

  'urea': { min: 0.5, max: 60, unit: 'mmol/L', notes: 'Blood urea nitrogen' },
  'blood urea': { min: 0.5, max: 60, unit: 'mmol/L' },
  'bun': { min: 0.5, max: 60, unit: 'mmol/L' },
  'blood urea nitrogen': { min: 0.5, max: 60, unit: 'mmol/L' },

  'uric acid': { min: 0.5, max: 15, unit: 'mg/dL', notes: 'Gout to tumor lysis syndrome' },
  'urate': { min: 0.5, max: 15, unit: 'mg/dL' },

  'microalbumin': { min: 0, max: 1000, unit: 'mg/L', notes: 'Microalbuminuria marker' },
  'microalbuminuria': { min: 0, max: 1000, unit: 'mg/L' },

  'acr': { min: 0, max: 500, unit: 'mg/mmol', notes: 'Albumin:creatinine ratio' },
  'albumin creatinine ratio': { min: 0, max: 500, unit: 'mg/mmol' },

  // ===================================
  // ELECTROLYTES & MINERALS
  // ===================================

  'sodium': { min: 100, max: 180, unit: 'mmol/L', notes: 'Survival range for severe hypo/hypernatremia' },
  'na': { min: 100, max: 180, unit: 'mmol/L' },
  'na+': { min: 100, max: 180, unit: 'mmol/L' },
  'serum sodium': { min: 100, max: 180, unit: 'mmol/L' },

  'potassium': { min: 1.5, max: 10, unit: 'mmol/L', notes: 'Cardiac arrest risk outside 2-8' },
  'k': { min: 1.5, max: 10, unit: 'mmol/L' },
  'k+': { min: 1.5, max: 10, unit: 'mmol/L' },
  'serum potassium': { min: 1.5, max: 10, unit: 'mmol/L' },

  'chloride': { min: 70, max: 140, unit: 'mmol/L' },
  'cl': { min: 70, max: 140, unit: 'mmol/L' },
  'cl-': { min: 70, max: 140, unit: 'mmol/L' },
  'serum chloride': { min: 70, max: 140, unit: 'mmol/L' },

  'bicarbonate': { min: 5, max: 50, unit: 'mmol/L', notes: 'Metabolic acidosis/alkalosis' },
  'hco3': { min: 5, max: 50, unit: 'mmol/L' },
  'co2': { min: 5, max: 50, unit: 'mmol/L' },
  'total co2': { min: 5, max: 50, unit: 'mmol/L' },

  'anion gap': { min: -5, max: 40, unit: 'mmol/L', notes: 'Calculated anion gap' },
  'anion-gap': { min: -5, max: 40, unit: 'mmol/L' },

  'calcium': { min: 1, max: 4, unit: 'mmol/L', notes: 'Hypocalcemia to hypercalcemia' },
  'ca': { min: 1, max: 4, unit: 'mmol/L' },
  'ca++': { min: 1, max: 4, unit: 'mmol/L' },
  'serum calcium': { min: 1, max: 4, unit: 'mmol/L' },
  'calcium total': { min: 1, max: 4, unit: 'mmol/L' },

  'magnesium': { min: 0.3, max: 5, unit: 'mmol/L' },
  'mg': { min: 0.3, max: 5, unit: 'mmol/L' },
  'serum magnesium': { min: 0.3, max: 5, unit: 'mmol/L' },

  'phosphate': { min: 0.3, max: 8, unit: 'mmol/L', notes: 'Tumor lysis syndrome' },
  'phosphorus': { min: 0.3, max: 8, unit: 'mmol/L' },
  'po4': { min: 0.3, max: 8, unit: 'mmol/L' },
  'inorganic phosphate': { min: 0.3, max: 8, unit: 'mmol/L' },

  // ===================================
  // THYROID FUNCTION
  // ===================================

  'tsh': { min: 0.001, max: 150, unit: 'mIU/L', notes: 'Suppressed to severe hypothyroidism' },
  'thyroid stimulating hormone': { min: 0.001, max: 150, unit: 'mIU/L' },
  'thyrotropin': { min: 0.001, max: 150, unit: 'mIU/L' },

  'free t4': { min: 1, max: 100, unit: 'pmol/L', notes: 'Free thyroxine' },
  'ft4': { min: 1, max: 100, unit: 'pmol/L' },
  'thyroxine free': { min: 1, max: 100, unit: 'pmol/L' },
  'free thyroxine': { min: 1, max: 100, unit: 'pmol/L' },

  'free t3': { min: 1, max: 30, unit: 'pmol/L', notes: 'Free triiodothyronine' },
  'ft3': { min: 1, max: 30, unit: 'pmol/L' },
  'triiodothyronine free': { min: 1, max: 30, unit: 'pmol/L' },
  'free triiodothyronine': { min: 1, max: 30, unit: 'pmol/L' },

  'reverse t3': { min: 5, max: 100, unit: 'ng/dL', notes: 'Reverse T3' },
  'reverse-t3': { min: 5, max: 100, unit: 'ng/dL' },
  'rt3': { min: 5, max: 100, unit: 'ng/dL' },

  'total t4': { min: 20, max: 300, unit: 'nmol/L', notes: 'Total thyroxine' },
  'total-t4': { min: 20, max: 300, unit: 'nmol/L' },
  't4': { min: 20, max: 300, unit: 'nmol/L' },

  'tpo antibodies': { min: 0, max: 5000, unit: 'IU/mL', notes: 'Thyroid peroxidase antibodies' },
  'tpo-antibodies': { min: 0, max: 5000, unit: 'IU/mL' },
  'anti-tpo': { min: 0, max: 5000, unit: 'IU/mL' },
  'thyroid peroxidase ab': { min: 0, max: 5000, unit: 'IU/mL' },

  'tg antibodies': { min: 0, max: 5000, unit: 'IU/mL', notes: 'Thyroglobulin antibodies' },
  'tg-antibodies': { min: 0, max: 5000, unit: 'IU/mL' },
  'anti-tg': { min: 0, max: 5000, unit: 'IU/mL' },
  'thyroglobulin ab': { min: 0, max: 5000, unit: 'IU/mL' },

  // ===================================
  // VITAMINS
  // ===================================

  '25oh vitamin d': { min: 5, max: 500, unit: 'nmol/L', notes: 'Toxicity symptoms start around 375' },
  'vitamin d': { min: 5, max: 500, unit: 'nmol/L' },
  '25-oh vitamin d': { min: 5, max: 500, unit: 'nmol/L' },
  'vitamin d 25-oh': { min: 5, max: 500, unit: 'nmol/L' },
  '25-hydroxyvitamin d': { min: 5, max: 500, unit: 'nmol/L' },
  '25 hydroxy vitamin d': { min: 5, max: 500, unit: 'nmol/L' },
  'vit d': { min: 5, max: 500, unit: 'nmol/L' },

  'vitamin b12': { min: 50, max: 10000, unit: 'pg/mL', notes: 'Can be very high with supplementation' },
  'b12': { min: 50, max: 10000, unit: 'pg/mL' },
  'cobalamin': { min: 50, max: 10000, unit: 'pg/mL' },
  'vit b12': { min: 50, max: 10000, unit: 'pg/mL' },

  'active b12': { min: 20, max: 500, unit: 'pmol/L', notes: 'Active B12 (holotranscobalamin)' },
  'active-b12': { min: 20, max: 500, unit: 'pmol/L' },
  'holotranscobalamin': { min: 20, max: 500, unit: 'pmol/L' },

  'folate': { min: 1, max: 100, unit: 'nmol/L', notes: 'Folic acid' },
  'folic acid': { min: 1, max: 100, unit: 'nmol/L' },
  'serum folate': { min: 1, max: 100, unit: 'nmol/L' },
  'vit b9': { min: 1, max: 100, unit: 'nmol/L' },

  'vitamin b6': { min: 5, max: 500, unit: 'nmol/L', notes: 'Pyridoxine' },
  'vitamin-b6': { min: 5, max: 500, unit: 'nmol/L' },
  'pyridoxine': { min: 5, max: 500, unit: 'nmol/L' },

  'homocysteine': { min: 1, max: 200, unit: 'µmol/L', notes: 'Elevated in B vitamin deficiency' },

  // ===================================
  // IRON STUDIES
  // ===================================

  'iron': { min: 5, max: 600, unit: 'µg/dL', notes: 'Severe iron overload rarely > 500' },
  'serum iron': { min: 5, max: 600, unit: 'µg/dL' },
  'fe': { min: 5, max: 600, unit: 'µg/dL' },

  'ferritin': { min: 1, max: 15000, unit: 'ng/mL', notes: 'Hemochromatosis can reach 10000+' },
  'serum ferritin': { min: 1, max: 15000, unit: 'ng/mL' },

  'tibc': { min: 100, max: 800, unit: 'µg/dL', notes: 'Total iron binding capacity' },
  'total iron binding capacity': { min: 100, max: 800, unit: 'µg/dL' },

  'transferrin': { min: 0.5, max: 6, unit: 'g/L' },
  'serum transferrin': { min: 0.5, max: 6, unit: 'g/L' },

  'transferrin saturation': { min: 1, max: 100, unit: '%', notes: 'Iron saturation' },
  'saturation': { min: 1, max: 100, unit: '%' },
  'iron saturation': { min: 1, max: 100, unit: '%' },
  'tsat': { min: 1, max: 100, unit: '%' },

  // ===================================
  // INFLAMMATION MARKERS
  // ===================================

  'crp': { min: 0, max: 500, unit: 'mg/L', notes: 'Severe sepsis can reach > 400' },
  'c-reactive protein': { min: 0, max: 500, unit: 'mg/L' },
  'c reactive protein': { min: 0, max: 500, unit: 'mg/L' },

  'hs-crp': { min: 0, max: 20, unit: 'mg/L', notes: 'High-sensitivity CRP' },
  'hs crp': { min: 0, max: 20, unit: 'mg/L' },
  'high sensitivity crp': { min: 0, max: 20, unit: 'mg/L' },

  'esr': { min: 0, max: 150, unit: 'mm/hr', notes: 'Erythrocyte sedimentation rate' },
  'erythrocyte sedimentation rate': { min: 0, max: 150, unit: 'mm/hr' },
  'sed rate': { min: 0, max: 150, unit: 'mm/hr' },

  'calprotectin': { min: 0, max: 2000, unit: 'µg/g', notes: 'Fecal calprotectin - IBD marker' },
  'fecal calprotectin': { min: 0, max: 2000, unit: 'µg/g' },

  // ===================================
  // HORMONES - MALE
  // ===================================

  'testosterone': { min: 0.1, max: 100, unit: 'nmol/L', notes: 'Total testosterone' },
  'total testosterone': { min: 0.1, max: 100, unit: 'nmol/L' },

  'free testosterone': { min: 0.1, max: 1000, unit: 'pmol/L' },

  'shbg': { min: 5, max: 200, unit: 'nmol/L', notes: 'Sex hormone binding globulin' },
  'sex hormone binding globulin': { min: 5, max: 200, unit: 'nmol/L' },

  'psa': { min: 0, max: 1000, unit: 'ng/mL', notes: 'Prostate specific antigen - cancer can be very high' },
  'prostate specific antigen': { min: 0, max: 1000, unit: 'ng/mL' },

  'dhea-s': { min: 10, max: 10000, unit: 'µg/dL', notes: 'DHEA sulfate' },
  'dhea sulfate': { min: 10, max: 10000, unit: 'µg/dL' },
  'dheas': { min: 10, max: 10000, unit: 'µg/dL' },

  // ===================================
  // HORMONES - FEMALE
  // ===================================

  'estradiol': { min: 10, max: 5000, unit: 'pmol/L', notes: 'E2 levels' },
  'estradiol-female': { min: 10, max: 5000, unit: 'pmol/L' },
  'e2': { min: 10, max: 5000, unit: 'pmol/L' },

  'progesterone': { min: 0.1, max: 200, unit: 'nmol/L', notes: 'Pregnancy levels' },

  'lh': { min: 0.1, max: 200, unit: 'IU/L', notes: 'Luteinizing hormone' },
  'luteinizing hormone': { min: 0.1, max: 200, unit: 'IU/L' },
  'lh-female': { min: 0.1, max: 200, unit: 'IU/L' },

  'fsh': { min: 0.1, max: 200, unit: 'IU/L', notes: 'Follicle stimulating hormone' },
  'follicle stimulating hormone': { min: 0.1, max: 200, unit: 'IU/L' },
  'fsh-female': { min: 0.1, max: 200, unit: 'IU/L' },

  'prolactin': { min: 1, max: 10000, unit: 'mIU/L', notes: 'Prolactinoma can cause very high levels' },
  'serum prolactin': { min: 1, max: 10000, unit: 'mIU/L' },

  'amh': { min: 0, max: 100, unit: 'pmol/L', notes: 'Anti-Müllerian hormone' },
  'anti-müllerian hormone': { min: 0, max: 100, unit: 'pmol/L' },
  'anti-mullerian hormone': { min: 0, max: 100, unit: 'pmol/L' },

  // ===================================
  // HORMONES - ADRENAL
  // ===================================

  'cortisol': { min: 10, max: 2000, unit: 'nmol/L', notes: 'Cushing\'s syndrome' },
  'serum cortisol': { min: 10, max: 2000, unit: 'nmol/L' },

  // ===================================
  // CARDIAC MARKERS
  // ===================================

  'troponin': { min: 0, max: 100000, unit: 'ng/L', notes: 'High-sensitivity troponin' },
  'troponin i': { min: 0, max: 100000, unit: 'ng/L' },
  'troponin t': { min: 0, max: 100000, unit: 'ng/L' },
  'hs-troponin': { min: 0, max: 100000, unit: 'ng/L' },

  'bnp': { min: 0, max: 10000, unit: 'pg/mL', notes: 'Brain natriuretic peptide' },
  'brain natriuretic peptide': { min: 0, max: 10000, unit: 'pg/mL' },

  'nt-probnp': { min: 0, max: 50000, unit: 'pg/mL', notes: 'NT-proBNP' },
  'nt-pro-bnp': { min: 0, max: 50000, unit: 'pg/mL' },
  'ntprobnp': { min: 0, max: 50000, unit: 'pg/mL' },

  'ck-mb': { min: 0, max: 1000, unit: 'U/L', notes: 'Creatine kinase MB' },
  'creatine kinase mb': { min: 0, max: 1000, unit: 'U/L' },

  'myoglobin': { min: 0, max: 5000, unit: 'ng/mL', notes: 'Rhabdomyolysis' },

  'lactate': { min: 0.1, max: 30, unit: 'mmol/L', notes: 'Lactic acidosis' },
  'lactic acid': { min: 0.1, max: 30, unit: 'mmol/L' },

  // ===================================
  // BLOOD GASES
  // ===================================

  'ph': { min: 6.8, max: 7.8, unit: '', notes: 'Life-threatening outside 7.0-7.7' },
  'blood ph': { min: 6.8, max: 7.8, unit: '' },

  'pco2': { min: 10, max: 120, unit: 'mmHg', notes: 'Partial pressure CO2' },
  'partial pressure co2': { min: 10, max: 120, unit: 'mmHg' },
  'carbon dioxide': { min: 10, max: 120, unit: 'mmHg' },

  'po2': { min: 20, max: 600, unit: 'mmHg', notes: 'Partial pressure O2 - can be very high on ventilation' },
  'partial pressure o2': { min: 20, max: 600, unit: 'mmHg' },
  'oxygen': { min: 20, max: 600, unit: 'mmHg' },

  'base excess': { min: -30, max: 30, unit: 'mmol/L', notes: 'Metabolic acidosis/alkalosis' },
  'base-excess': { min: -30, max: 30, unit: 'mmol/L' },

  'oxygen saturation': { min: 40, max: 100, unit: '%', notes: 'SpO2' },
  'sao2': { min: 40, max: 100, unit: '%' },
  'o2 saturation': { min: 40, max: 100, unit: '%' },

  // ===================================
  // COAGULATION
  // ===================================

  'pt': { min: 5, max: 120, unit: 's', notes: 'Prothrombin time' },
  'prothrombin time': { min: 5, max: 120, unit: 's' },

  'inr': { min: 0.5, max: 20, unit: 'ratio', notes: 'International normalized ratio - warfarin therapy' },
  'international normalized ratio': { min: 0.5, max: 20, unit: 'ratio' },

  'aptt': { min: 10, max: 200, unit: 's', notes: 'Activated partial thromboplastin time' },
  'activated partial thromboplastin time': { min: 10, max: 200, unit: 's' },
  'ptt': { min: 10, max: 200, unit: 's' },

  'd-dimer': { min: 0, max: 20000, unit: 'ng/mL', notes: 'Elevated in thrombosis, DIC' },
  'd dimer': { min: 0, max: 20000, unit: 'ng/mL' },

  'fibrinogen': { min: 0.5, max: 10, unit: 'g/L', notes: 'Coagulation factor' },

  // ===================================
  // BONE HEALTH
  // ===================================

  'pth': { min: 1, max: 500, unit: 'pg/mL', notes: 'Parathyroid hormone' },
  'parathyroid hormone': { min: 1, max: 500, unit: 'pg/mL' },

  'osteocalcin': { min: 1, max: 200, unit: 'ng/mL', notes: 'Bone formation marker' },
  'bone osteocalcin': { min: 1, max: 200, unit: 'ng/mL' },

  'p1np': { min: 5, max: 500, unit: 'ng/mL', notes: 'Bone formation (P1NP)' },
  'procollagen type 1 n-terminal propeptide': { min: 5, max: 500, unit: 'ng/mL' },

  'ctx': { min: 10, max: 2000, unit: 'pg/mL', notes: 'Bone resorption (CTX)' },
  'c-telopeptide': { min: 10, max: 2000, unit: 'pg/mL' },

  // ===================================
  // AUTOIMMUNE & IMMUNOLOGY
  // ===================================

  'ana': { min: 0, max: 1000, unit: 'titre', notes: 'Antinuclear antibodies titre' },
  'antinuclear antibodies': { min: 0, max: 1000, unit: 'titre' },
  'anti-nuclear ab': { min: 0, max: 1000, unit: 'titre' },

  'anti-dsdna': { min: 0, max: 1000, unit: 'IU/mL', notes: 'Anti-double stranded DNA (SLE)' },
  'anti-ds-dna': { min: 0, max: 1000, unit: 'IU/mL' },

  'rheumatoid factor': { min: 0, max: 1000, unit: 'IU/mL', notes: 'RF for rheumatoid arthritis' },
  'rf': { min: 0, max: 1000, unit: 'IU/mL' },

  'anti-ccp': { min: 0, max: 1000, unit: 'U/mL', notes: 'Anti-cyclic citrullinated peptide' },
  'anti-ccp antibodies': { min: 0, max: 1000, unit: 'U/mL' },
  'anti-cyclic citrullinated peptide': { min: 0, max: 1000, unit: 'U/mL' },

  'complement c3': { min: 0.2, max: 3, unit: 'g/L', notes: 'Complement component C3' },
  'complement-c3': { min: 0.2, max: 3, unit: 'g/L' },
  'c3': { min: 0.2, max: 3, unit: 'g/L' },

  'complement c4': { min: 0.05, max: 1, unit: 'g/L', notes: 'Complement component C4' },
  'complement-c4': { min: 0.05, max: 1, unit: 'g/L' },
  'c4': { min: 0.05, max: 1, unit: 'g/L' },

  'total ige': { min: 1, max: 10000, unit: 'kU/L', notes: 'Total IgE - allergies/parasites' },
  'total-ige': { min: 1, max: 10000, unit: 'kU/L' },
  'ige': { min: 1, max: 10000, unit: 'kU/L' },

  // ===================================
  // TUMOR MARKERS
  // ===================================

  'cea': { min: 0, max: 10000, unit: 'ng/mL', notes: 'Carcinoembryonic antigen - GI cancers' },
  'carcinoembryonic antigen': { min: 0, max: 10000, unit: 'ng/mL' },

  'ca125': { min: 0, max: 10000, unit: 'U/mL', notes: 'Ovarian cancer marker' },
  'ca 125': { min: 0, max: 10000, unit: 'U/mL' },

  'ca19-9': { min: 0, max: 10000, unit: 'U/mL', notes: 'Pancreatic cancer marker' },
  'ca 19-9': { min: 0, max: 10000, unit: 'U/mL' },

  'afp': { min: 0, max: 100000, unit: 'ng/mL', notes: 'Alpha-fetoprotein - liver/germ cell tumors' },
  'alpha-fetoprotein': { min: 0, max: 100000, unit: 'ng/mL' },
  'alpha fetoprotein': { min: 0, max: 100000, unit: 'ng/mL' },

  // ===================================
  // DRUG MONITORING
  // ===================================

  'lithium': { min: 0, max: 5, unit: 'mmol/L', notes: 'Therapeutic 0.6-1.2, toxic > 1.5' },
  'lithium level': { min: 0, max: 5, unit: 'mmol/L' },
  'serum lithium': { min: 0, max: 5, unit: 'mmol/L' },

  'digoxin': { min: 0, max: 10, unit: 'ng/mL', notes: 'Therapeutic 0.8-2.0, toxic > 2.5' },
  'digoxin level': { min: 0, max: 10, unit: 'ng/mL' },

  'phenytoin': { min: 0, max: 100, unit: 'mg/L', notes: 'Therapeutic 10-20' },
  'dilantin': { min: 0, max: 100, unit: 'mg/L' },

  'valproic acid': { min: 0, max: 300, unit: 'mg/L', notes: 'Therapeutic 50-100' },
  'valproate': { min: 0, max: 300, unit: 'mg/L' },

  'carbamazepine': { min: 0, max: 50, unit: 'mg/L', notes: 'Therapeutic 4-12' },
  'tegretol': { min: 0, max: 50, unit: 'mg/L' },

  'vancomycin': { min: 0, max: 100, unit: 'mg/L', notes: 'Therapeutic trough 10-20' },
  'vanco level': { min: 0, max: 100, unit: 'mg/L' },
  'vancomycin trough': { min: 0, max: 100, unit: 'mg/L' },

  'gentamicin': { min: 0, max: 40, unit: 'mg/L', notes: 'Therapeutic peak < 10, trough < 2' },
  'gentamicin level': { min: 0, max: 40, unit: 'mg/L' },

  'methotrexate': { min: 0, max: 1000, unit: 'µmol/L', notes: 'Chemotherapy monitoring' },
  'mtx': { min: 0, max: 1000, unit: 'µmol/L' },

  // ===================================
  // HEAVY METALS & TOXICOLOGY
  // ===================================

  'lead': { min: 0, max: 500, unit: 'µg/dL', notes: 'Lead toxicity' },
  'blood lead': { min: 0, max: 500, unit: 'µg/dL' },

  'mercury': { min: 0, max: 200, unit: 'µg/L', notes: 'Mercury toxicity' },
  'blood mercury': { min: 0, max: 200, unit: 'µg/L' },

  'copper': { min: 5, max: 500, unit: 'µg/dL', notes: 'Wilson\'s disease can be very low' },
  'serum copper': { min: 5, max: 500, unit: 'µg/dL' },

  'ceruloplasmin': { min: 1, max: 100, unit: 'mg/dL', notes: 'Copper-binding protein' },

  // ===================================
  // INFECTIOUS DISEASE
  // ===================================

  'hiv-antibody': { min: 0, max: 1, unit: 'index', notes: 'Reactive/non-reactive' },
  'hiv antibody': { min: 0, max: 1, unit: 'index' },

  'hbsag': { min: 0, max: 1000, unit: 'IU/mL', notes: 'Hepatitis B surface antigen' },
  'hepatitis b surface antigen': { min: 0, max: 1000, unit: 'IU/mL' },

  'hcv-antibody': { min: 0, max: 100, unit: 'S/CO', notes: 'Hepatitis C antibody' },
  'hepatitis c antibody': { min: 0, max: 100, unit: 'S/CO' },

  'anti-ttg-iga': { min: 0, max: 500, unit: 'U/mL', notes: 'Anti-tissue transglutaminase (celiac)' },
  'anti-ttg': { min: 0, max: 500, unit: 'U/mL' },
  'tissue transglutaminase ab': { min: 0, max: 500, unit: 'U/mL' },

  'h-pylori-ab': { min: 0, max: 100, unit: 'U/mL', notes: 'H. pylori antibodies' },
  'h pylori antibody': { min: 0, max: 100, unit: 'U/mL' },
  'helicobacter pylori ab': { min: 0, max: 100, unit: 'U/mL' },

  // ===================================
  // OTHER SPECIALIZED
  // ===================================

  'zonulin': { min: 0, max: 200, unit: 'ng/mL', notes: 'Intestinal permeability marker' },
  'serum zonulin': { min: 0, max: 200, unit: 'ng/mL' },

  'alpha-1-antitrypsin': { min: 20, max: 500, unit: 'mg/dL', notes: 'Alpha-1 antitrypsin deficiency' },
  'alpha-1 antitrypsin': { min: 20, max: 500, unit: 'mg/dL' },
  'a1at': { min: 20, max: 500, unit: 'mg/dL' },
};

/**
 * Normalize biomarker name for lookup
 */
export function normalizeBiomarkerName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Check if a biomarker value is biologically plausible
 */
export function checkBiologicalPlausibility(
  biomarkerName: string,
  value: number
): { isPlausible: boolean; issue?: string } {
  const normalizedName = normalizeBiomarkerName(biomarkerName);

  // Check for exact match first
  let limits = BIOLOGICAL_PLAUSIBILITY_LIMITS[normalizedName];

  if (!limits) {
    // Try partial matching for compound names (e.g., "LDL Cholesterol Calculated" -> "ldl cholesterol")
    for (const [key, val] of Object.entries(BIOLOGICAL_PLAUSIBILITY_LIMITS)) {
      if (normalizedName.includes(key) || key.includes(normalizedName)) {
        limits = val;
        break;
      }
    }
  }

  // If still no match, assume plausible (let reference range validation catch it)
  if (!limits) {
    return { isPlausible: true };
  }

  // Check if value is within plausible range
  if (value < limits.min || value > limits.max) {
    return {
      isPlausible: false,
      issue: `Value ${value} ${limits.unit} outside plausible range ${limits.min}-${limits.max} ${limits.unit}`,
    };
  }

  return { isPlausible: true };
}

/**
 * Get plausibility limits for a biomarker (for display/debugging)
 */
export function getPlausibilityLimits(biomarkerName: string): PlausibilityLimit | undefined {
  const normalizedName = normalizeBiomarkerName(biomarkerName);

  let limits = BIOLOGICAL_PLAUSIBILITY_LIMITS[normalizedName];

  if (!limits) {
    for (const [key, val] of Object.entries(BIOLOGICAL_PLAUSIBILITY_LIMITS)) {
      if (normalizedName.includes(key) || key.includes(normalizedName)) {
        limits = val;
        break;
      }
    }
  }

  return limits;
}

/**
 * Get count of biomarkers with validation rules
 */
export function getValidationCoverage(): { total: number; byCategory: Record<string, number> } {
  const uniqueKeys = new Set(Object.keys(BIOLOGICAL_PLAUSIBILITY_LIMITS));

  const categories: Record<string, number> = {
    haematology: 0,
    metabolic: 0,
    lipids: 0,
    liver: 0,
    kidney: 0,
    electrolytes: 0,
    thyroid: 0,
    vitamins: 0,
    iron: 0,
    inflammation: 0,
    hormones: 0,
    cardiac: 0,
    bloodgases: 0,
    coagulation: 0,
    bone: 0,
    autoimmune: 0,
    tumormarkers: 0,
    drugmonitoring: 0,
    toxicology: 0,
    infectious: 0,
    other: 0,
  };

  // Categorize based on biomarker names
  for (const key of uniqueKeys) {
    if (key.includes('haemoglobin') || key.includes('haematocrit') || key.includes('platelet') || key.includes('white cell') || key.includes('red cell') || key.includes('neutrophil') || key.includes('lymphocyte')) {
      categories.haematology++;
    } else if (key.includes('glucose') || key.includes('hba1c') || key.includes('insulin')) {
      categories.metabolic++;
    } else if (key.includes('cholesterol') || key.includes('triglyceride') || key.includes('hdl') || key.includes('ldl') || key.includes('apob')) {
      categories.lipids++;
    } else if (key.includes('alt') || key.includes('ast') || key.includes('ggt') || key.includes('alp') || key.includes('bilirubin') || key.includes('albumin')) {
      categories.liver++;
    } else if (key.includes('creatinine') || key.includes('egfr') || key.includes('urea') || key.includes('cystatin')) {
      categories.kidney++;
    } else if (key.includes('sodium') || key.includes('potassium') || key.includes('chloride') || key.includes('bicarbonate') || key.includes('calcium') || key.includes('magnesium') || key.includes('phosphate')) {
      categories.electrolytes++;
    } else if (key.includes('tsh') || key.includes('free t') || key.includes('thyroid') || key.includes('tpo')) {
      categories.thyroid++;
    } else if (key.includes('vitamin') || key.includes('folate') || key.includes('homocysteine')) {
      categories.vitamins++;
    } else if (key.includes('iron') || key.includes('ferritin') || key.includes('tibc') || key.includes('transferrin')) {
      categories.iron++;
    } else if (key.includes('crp') || key.includes('esr') || key.includes('calprotectin')) {
      categories.inflammation++;
    } else if (key.includes('testosterone') || key.includes('estradiol') || key.includes('progesterone') || key.includes('lh') || key.includes('fsh') || key.includes('prolactin') || key.includes('cortisol') || key.includes('psa') || key.includes('dhea')) {
      categories.hormones++;
    } else if (key.includes('troponin') || key.includes('bnp') || key.includes('lactate')) {
      categories.cardiac++;
    } else if (key.includes('ph') || key.includes('pco2') || key.includes('po2') || key.includes('base excess') || key.includes('oxygen')) {
      categories.bloodgases++;
    } else if (key.includes('pt') || key.includes('inr') || key.includes('aptt') || key.includes('d-dimer') || key.includes('fibrinogen')) {
      categories.coagulation++;
    } else if (key.includes('pth') || key.includes('osteocalcin') || key.includes('p1np') || key.includes('ctx')) {
      categories.bone++;
    } else if (key.includes('ana') || key.includes('anti-') || key.includes('rheumatoid') || key.includes('complement')) {
      categories.autoimmune++;
    } else if (key.includes('cea') || key.includes('ca125') || key.includes('ca19') || key.includes('afp')) {
      categories.tumormarkers++;
    } else if (key.includes('lithium') || key.includes('digoxin') || key.includes('phenytoin') || key.includes('valproic') || key.includes('vancomycin') || key.includes('gentamicin')) {
      categories.drugmonitoring++;
    } else if (key.includes('lead') || key.includes('mercury') || key.includes('copper') || key.includes('ceruloplasmin')) {
      categories.toxicology++;
    } else if (key.includes('hiv') || key.includes('hbsag') || key.includes('hcv') || key.includes('anti-ttg') || key.includes('h-pylori') || key.includes('h pylori')) {
      categories.infectious++;
    } else {
      categories.other++;
    }
  }

  return {
    total: uniqueKeys.size,
    byCategory: categories,
  };
}
