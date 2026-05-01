# Technical Write-Up

## 1. VLM choice and alternatives

I selected `Qwen/Qwen2.5-VL-3B-Instruct` as the main model and `HuggingFaceTB/SmolVLM2-2.2B-Instruct` as the comparison baseline.

Why Qwen2.5-VL-3B:

- It is open-source and within the assignment limit of 13B parameters.
- Qwen explicitly highlights document parsing, tables, forms, OCR-heavy understanding, and structured extraction in both its release material and technical report.
- The 3B checkpoint is a practical middle ground: noticeably stronger than very small models, but still realistic for local experiments or free Colab.
- For this assignment, output format discipline matters. Qwen is a better fit when the prompt asks for strict JSON.

Why SmolVLM2-2.2B as the baseline:

- It is a lightweight open-source multimodal model with good document VQA quality for its size.
- It gives a fair efficiency baseline against Qwen instead of comparing two large models.
- It is easier to run when memory is tight.

Alternatives considered:

- Donut fine-tunes for receipts: strong on narrow receipt parsing, but weaker as a single general solution for signatures, forms, and broader document understanding.
- LLaVA-style models: capable, but not my first choice for document-heavy extraction where OCR and layout understanding matter more.

## 2. Prompting strategy

The best prompt pattern for this assignment was strict structured prompting with implicit reasoning:

- one shared system instruction saying “return valid JSON only”
- one user prompt with a fixed task-specific output schema
- explicit rules for missing values (`null`) and no markdown fences
- internal reasoning only, with the final answer restricted to JSON

This was better than verbose prompts or explicit chain-of-thought because those approaches increased formatting mistakes and made evaluation less stable. Fixed schemas made evaluation easier, and keeping reasoning implicit preserved clean machine-readable outputs.

In practice, I do not use one single universal prompt for every task. The code uses:

- one shared system prompt for all document extraction tasks
- multiple different user prompts, one per task
- a fixed schema for each task so the output format matches the downstream evaluator or demo use case
- a repair pass that is triggered if the first model response is invalid JSON or does not match the expected top-level schema

The task-specific prompts cover:

- generic full-document extraction
- key-value extraction
- signature detection
- form-field status extraction
- receipt summary extraction
- CORD-specific receipt extraction

For the more general document reasoning part of the task, I found that the best-performing prompt style was not full visible chain-of-thought. Instead, the better pattern was:

- ask the model to reason internally about document type, entities, and field relationships
- force the final output to stay within a strict task schema
- run a second schema-repair prompt only when the first output is malformed

This gives better practical results for document question answering in this repository because the downstream objective is accurate structured extraction, not long-form explanation quality.

## 3. Pipeline design

The pipeline is centered around one main runner and three core modules:

1. `run_pipeline.py` for end-to-end orchestration
2. `pdf_to_image.py` for rendering and preprocessing
3. `extract.py` for VLM inference
4. `evaluate.py` for metrics

Preprocessing is intentionally light:

- render each PDF page separately
- deskew
- denoise
- grayscale
- resize to a reasonable max side

I kept preprocessing conservative because aggressive cleaning can damage small text or signatures.

Operationally, the flow is:

- `run_pipeline.py` loads config, selects the model, and iterates over PDF or image inputs
- `pdf_to_image.py` renders PDF pages with PyMuPDF and preprocesses them
- `extract.py` applies the task-specific prompt and returns structured JSON per page
- `evaluate.py` is called only when ground-truth JSON is available
- for benchmarking, `prepare_cord.py` prepares the CORD subset and `model_compare.py` runs both selected VLMs on the same data

## 4. Evaluation approach

For quantitative benchmarking, I use the real CORD dataset instead of any local synthetic labels.

How I use CORD:

- `prepare_cord.py` downloads a chosen subset from `naver-clova-ix/cord-v2`
- the original CORD `ground_truth` is normalized into a simpler target with `menu`, `sub_total`, and `total`
- both models are run on the same CORD images through `model_compare.py`
- `prepare_cord.py` writes a `metadata.csv` file so the exact evaluated subset is recorded
- metrics are computed from flattened JSON path-value pairs

For structured fields:

- Exact Match for direct field equality
- Precision / Recall / F1 over normalized key-value matches

CORD does not provide signature labels, so signature metrics are not part of the CORD benchmark itself. Signature detection remains in the generic document pipeline for PDF demos and can be extended later with a signature-labeled dataset.

For PDF testing:

- I generate a small set of realistic PDF documents with `reportlab`
- these PDFs are not used for benchmark metrics
- they are only used to validate PDF rendering, preprocessing, and structured inference on true PDF input
- I also save the source metadata for each generated PDF so the demo inputs are explainable during the interview

## 5. Failure cases and limitations

- Long dense tables may still be partially summarized instead of fully extracted.
- Signature detection still needs a dedicated labeled dataset for serious measurement.
- Generic VLMs may hallucinate missing fields if the prompt is too open.
- The LoRA script is provided as a practical starter, but full tuning should be run on GPU or Colab.

## 6. Production improvements

- Add a document classifier first and route receipts, forms, and agreements to specialized prompts.
- Fine-tune on a mixed set of receipts plus signature/form examples.
- Add confidence scoring and human-review fallback for low-confidence fields.
- Move batch inference to vLLM on Linux/CUDA for real throughput.
- Store prompt versions and model outputs for easier auditing.
