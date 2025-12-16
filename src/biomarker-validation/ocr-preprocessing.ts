/**
 * OCR Preprocessing Pipeline for Enhanced Accuracy
 *
 * Implements image enhancement techniques before Tesseract OCR:
 * - Denoising
 * - Binarization (adaptive thresholding)
 * - Deskewing
 * - Contrast enhancement
 * - Multi-pass OCR with confidence thresholding
 *
 * Target: 98%+ OCR accuracy on scanned PDFs
 */

export interface OCRPreprocessingOptions {
  denoise?: boolean;
  binarize?: boolean;
  deskew?: boolean;
  enhanceContrast?: boolean;
  scale?: number; // DPI multiplier (2.0 = 2x resolution)
  multiPass?: boolean; // Try multiple settings if confidence low
  confidenceThreshold?: number; // 0-100, retry if below threshold
}

export interface OCRResult {
  text: string;
  confidence: number;
  preprocessingApplied: string[];
  passes: number;
}

/**
 * Denoise image using median filter
 */
function denoiseCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Simple median filter (3x3 kernel)
  const newData = new Uint8ClampedArray(data.length);
  const width = canvas.width;
  const height = canvas.height;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = (y * width + x) * 4;

      // Collect surrounding pixels for median calculation
      const neighbors: number[] = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nIdx = ((y + dy) * width + (x + dx)) * 4;
          neighbors.push(data[nIdx]); // Grayscale value
        }
      }

      neighbors.sort((a, b) => a - b);
      const median = neighbors[Math.floor(neighbors.length / 2)];

      newData[idx] = median;
      newData[idx + 1] = median;
      newData[idx + 2] = median;
      newData[idx + 3] = 255; // Alpha
    }
  }

  const newImageData = new ImageData(newData, width, height);
  ctx.putImageData(newImageData, 0, 0);

  return canvas;
}

/**
 * Binarize image using Otsu's method (adaptive thresholding)
 */
function binarizeCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Convert to grayscale and calculate histogram
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    histogram[gray]++;
  }

  // Calculate threshold using Otsu's method
  const total = canvas.width * canvas.height;
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }

  let sumB = 0;
  let wB = 0;
  let wF = 0;
  let maxVariance = 0;
  let threshold = 0;

  for (let i = 0; i < 256; i++) {
    wB += histogram[i];
    if (wB === 0) continue;

    wF = total - wB;
    if (wF === 0) break;

    sumB += i * histogram[i];

    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;

    const variance = wB * wF * (mB - mF) * (mB - mF);

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = i;
    }
  }

  // Apply threshold
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    const binary = gray > threshold ? 255 : 0;
    data[i] = binary;
    data[i + 1] = binary;
    data[i + 2] = binary;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Detect and correct image skew
 */
function deskewCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;

  // Simple Hough transform for line detection
  // For production, consider using a library like OpenCV.js
  // This is a simplified version

  // Detect dominant angle (simplified)
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Edge detection (simple Sobel)
  const edges: { x: number; y: number }[] = [];
  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < canvas.width - 1; x++) {
      const idx = (y * canvas.width + x) * 4;
      const gray = data[idx];

      // Sobel operators
      const gx =
        -data[((y - 1) * canvas.width + (x - 1)) * 4] +
        data[((y - 1) * canvas.width + (x + 1)) * 4] -
        2 * data[(y * canvas.width + (x - 1)) * 4] +
        2 * data[(y * canvas.width + (x + 1)) * 4] -
        data[((y + 1) * canvas.width + (x - 1)) * 4] +
        data[((y + 1) * canvas.width + (x + 1)) * 4];

      const gy =
        -data[((y - 1) * canvas.width + (x - 1)) * 4] -
        2 * data[((y - 1) * canvas.width + x) * 4] -
        data[((y - 1) * canvas.width + (x + 1)) * 4] +
        data[((y + 1) * canvas.width + (x - 1)) * 4] +
        2 * data[((y + 1) * canvas.width + x) * 4] +
        data[((y + 1) * canvas.width + (x + 1)) * 4];

      const magnitude = Math.sqrt(gx * gx + gy * gy);

      if (magnitude > 50) {
        // Edge threshold
        edges.push({ x, y });
      }
    }
  }

  // Calculate dominant angle (simplified - would use Hough transform in production)
  // For now, assume image is mostly straight (skip rotation)
  // Full implementation would detect skew angle and rotate accordingly

  return canvas; // Placeholder - full deskewing requires more complex algorithm
}

/**
 * Enhance contrast using histogram equalization
 */
function enhanceContrastCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Calculate histogram
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    histogram[gray]++;
  }

  // Calculate cumulative distribution
  const cdf = new Array(256).fill(0);
  cdf[0] = histogram[0];
  for (let i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + histogram[i];
  }

  // Normalize CDF
  const total = canvas.width * canvas.height;
  const cdfMin = cdf.find((v) => v > 0) || 0;
  const equalizedMap = new Array(256);

  for (let i = 0; i < 256; i++) {
    equalizedMap[i] = Math.round(((cdf[i] - cdfMin) / (total - cdfMin)) * 255);
  }

  // Apply equalization
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    const enhanced = equalizedMap[gray];
    data[i] = enhanced;
    data[i + 1] = enhanced;
    data[i + 2] = enhanced;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Preprocess canvas before OCR
 */
export function preprocessCanvas(
  canvas: HTMLCanvasElement,
  options: OCRPreprocessingOptions = {}
): { canvas: HTMLCanvasElement; applied: string[] } {
  const {
    denoise = true,
    binarize = true,
    deskew = false, // Disabled by default (complex)
    enhanceContrast = true,
    scale = 2.0,
  } = options;

  const applied: string[] = [];

  // Create a copy of the canvas
  const processedCanvas = document.createElement('canvas');
  processedCanvas.width = canvas.width;
  processedCanvas.height = canvas.height;
  const ctx = processedCanvas.getContext('2d')!;
  ctx.drawImage(canvas, 0, 0);

  // Apply preprocessing steps in order
  if (enhanceContrast) {
    enhanceContrastCanvas(processedCanvas);
    applied.push('contrast_enhancement');
  }

  if (denoise) {
    denoiseCanvas(processedCanvas);
    applied.push('denoising');
  }

  if (binarize) {
    binarizeCanvas(processedCanvas);
    applied.push('binarization');
  }

  if (deskew) {
    deskewCanvas(processedCanvas);
    applied.push('deskewing');
  }

  return { canvas: processedCanvas, applied };
}

/**
 * Perform OCR with preprocessing and multi-pass support
 */
export async function performEnhancedOCR(
  canvas: HTMLCanvasElement,
  tesseractWorker: any, // Tesseract worker instance
  options: OCRPreprocessingOptions = {}
): Promise<OCRResult> {
  const {
    multiPass = true,
    confidenceThreshold = 80,
    ...preprocessOptions
  } = options;

  const results: Array<{ text: string; confidence: number; preprocessing: string[] }> = [];

  // Pass 1: Standard preprocessing
  const { canvas: processedCanvas1, applied: applied1 } = preprocessCanvas(canvas, preprocessOptions);

  const result1 = await tesseractWorker.recognize(processedCanvas1);
  const confidence1 = result1.data.confidence || 0;
  results.push({
    text: result1.data.text,
    confidence: confidence1,
    preprocessing: applied1,
  });

  console.log(`OCR Pass 1: ${confidence1.toFixed(1)}% confidence`);

  // If confidence is high enough, return immediately
  if (confidence1 >= confidenceThreshold || !multiPass) {
    return {
      text: result1.data.text,
      confidence: confidence1,
      preprocessingApplied: applied1,
      passes: 1,
    };
  }

  // Pass 2: Try with different preprocessing (no binarization)
  const { canvas: processedCanvas2, applied: applied2 } = preprocessCanvas(canvas, {
    ...preprocessOptions,
    binarize: false,
    enhanceContrast: true,
  });

  const result2 = await tesseractWorker.recognize(processedCanvas2);
  const confidence2 = result2.data.confidence || 0;
  results.push({
    text: result2.data.text,
    confidence: confidence2,
    preprocessing: applied2,
  });

  console.log(`OCR Pass 2: ${confidence2.toFixed(1)}% confidence`);

  if (confidence2 >= confidenceThreshold) {
    return {
      text: result2.data.text,
      confidence: confidence2,
      preprocessingApplied: applied2,
      passes: 2,
    };
  }

  // Pass 3: Try with original (no preprocessing except scaling)
  const result3 = await tesseractWorker.recognize(canvas);
  const confidence3 = result3.data.confidence || 0;
  results.push({
    text: result3.data.text,
    confidence: confidence3,
    preprocessing: ['none'],
  });

  console.log(`OCR Pass 3: ${confidence3.toFixed(1)}% confidence (no preprocessing)`);

  // Return best result
  const best = results.reduce((prev, current) =>
    current.confidence > prev.confidence ? current : prev
  );

  return {
    text: best.text,
    confidence: best.confidence,
    preprocessingApplied: best.preprocessing,
    passes: results.length,
  };
}

/**
 * Post-OCR text correction using biomarker vocabulary
 */
export function correctOCRErrors(text: string, biomarkerVocab: string[]): string {
  // Common OCR errors
  const ocrSubstitutions: Record<string, string> = {
    '0': 'O',
    'O': '0',
    '1': 'I',
    'I': '1',
    '5': 'S',
    'S': '5',
    '8': 'B',
    'B': '8',
    'l': '1',
    'Z': '2',
  };

  let corrected = text;

  // Find potential biomarker names in text
  const words = text.split(/\s+/);

  for (let i = 0; i < words.length; i++) {
    const word = words[i];

    // Check if word is close to any biomarker name
    for (const biomarker of biomarkerVocab) {
      if (levenshteinDistance(word.toLowerCase(), biomarker.toLowerCase()) <= 2) {
        // Close match - likely OCR error
        words[i] = biomarker;
        break;
      }
    }
  }

  corrected = words.join(' ');

  return corrected;
}

/**
 * Calculate Levenshtein distance (edit distance) between two strings
 */
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;

  const dp: number[][] = Array(len1 + 1)
    .fill(null)
    .map(() => Array(len2 + 1).fill(0));

  for (let i = 0; i <= len1; i++) {
    dp[i][0] = i;
  }

  for (let j = 0; j <= len2; j++) {
    dp[0][j] = j;
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (str1[i - 1] === str2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  return dp[len1][len2];
}

/**
 * Extract biomarker names from database for OCR correction
 */
export function getBiomarkerVocabulary(): string[] {
  // This would be imported from biomarkersExpanded.ts in production
  // For now, return common biomarkers
  return [
    'Hemoglobin',
    'Haemoglobin',
    'Cholesterol',
    'Glucose',
    'Creatinine',
    'Sodium',
    'Potassium',
    'Chloride',
    'Bicarbonate',
    'TSH',
    'Free T4',
    'Triglycerides',
    'HDL',
    'LDL',
    'Platelets',
    'WBC',
    'RBC',
    'Ferritin',
    'Iron',
    'Vitamin D',
    'Vitamin B12',
    'Folate',
    'ALT',
    'AST',
    'GGT',
    'ALP',
    'Bilirubin',
    'Albumin',
    'Urea',
    'BUN',
    'eGFR',
    'HbA1c',
    'PSA',
    'Testosterone',
    'Estradiol',
    'Cortisol',
    'Prolactin',
    'LH',
    'FSH',
    'SHBG',
    'Calcium',
    'Magnesium',
    'Phosphate',
    'Uric Acid',
    'CRP',
    'ESR',
    'Troponin',
    'BNP',
    'pH',
    'pCO2',
    'pO2',
    'Base Excess',
    'Lithium',
    'Vancomycin',
    'Digoxin',
    'Tacrolimus',
    'Cyclosporine',
    'CA 19-9',
    'CA 125',
    'CEA',
    'AFP',
    'ANA',
    'Rheumatoid Factor',
    'Anti-CCP',
  ];
}
