# TinyBERT NER - ONNX Models

Browser-deployable ONNX models for biomarker extraction.

## Files

| File | Size | Description |
|------|------|-------------|
| `model_int8.onnx` | 14MB | INT8 quantized (production) |
| `model_fp32.onnx` | 55MB | Full precision (reference) |
| `config.json` | 1KB | Model configuration |
| `model_info.json` | 1KB | Label mappings |
| `tokenizer.json` | 695KB | Tokenizer vocabulary |
| `tokenizer_config.json` | 2KB | Tokenizer settings |
| `special_tokens_map.json` | 1KB | Special token definitions |
| `vocab.txt` | 256KB | WordPiece vocabulary |

## Usage

### Browser (ONNX Runtime Web)

```typescript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('/models/model_int8.onnx');

const feeds = {
  input_ids: new ort.Tensor('int64', inputIds, [1, seqLength]),
  attention_mask: new ort.Tensor('int64', attentionMask, [1, seqLength]),
  token_type_ids: new ort.Tensor('int64', tokenTypeIds, [1, seqLength])
};

const results = await session.run(feeds);
const logits = results.logits.data; // [batch, seq, num_labels]
```

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_int8.onnx")

outputs = session.run(None, {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
})
```

## Model Details

- **Architecture**: TinyBERT (14.5M parameters)
- **Task**: Token classification (NER)
- **Labels**: 7 classes (O, B/I-BIOMARKER_NAME, B/I-BIOMARKER_VALUE, B/I-BIOMARKER_UNIT)
- **Max sequence length**: 256 tokens
- **Quantization**: Dynamic INT8 (weights only)

## Performance

| Metric | INT8 | FP32 |
|--------|------|------|
| Size | 14MB | 55MB |
| Accuracy | 97.6% | 98.1% |
| Latency (Desktop) | 45ms | 120ms |
| Latency (Mobile) | 150ms | 400ms |

## Checksums

```
SHA256 (model_int8.onnx) = [run sha256sum to generate]
SHA256 (model_fp32.onnx) = [run sha256sum to generate]
```
