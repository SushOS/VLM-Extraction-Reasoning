# Technical Write-Up

## 1. Model selection

I selected `Qwen/Qwen2.5-VL-3B-Instruct` as the primary extraction model and `HuggingFaceTB/SmolVLM2-2.2B-Instruct` as the comparison baseline.

The technical reason for choosing Qwen2.5-VL-3B is that this assignment is a structured multimodal extraction problem, not just OCR. The model must jointly handle text recognition, local visual cues, instruction following, and schema-constrained generation. Those requirements are especially important for cases like:

- distinguishing `total` from `subtotal`
- preserving form-field state rather than summarizing the page
- detecting whether a visible signature mark exists
- returning valid JSON consistently enough for automated evaluation

Qwen2.5-VL-3B is a strong fit because it targets document-style vision-language tasks while staying comfortably below the 13B constraint. The 3B size is also a practical middle point: large enough to support better structured extraction behavior than very small multimodal models, but still realistic for local or Colab-style experimentation.

I selected SmolVLM2-2.2B as the baseline for a different reason. A good baseline should probe a different operational tradeoff, not merely be a weaker version of the same model family. SmolVLM2 is meaningfully smaller and cheaper to run, so it serves as a useful efficiency reference. That comparison makes the final result more informative: it helps answer whether the extra capacity of Qwen materially improves extraction quality enough to justify its higher compute cost.

I did not center the project on Donut-style or narrowly receipt-specialized models because the assignment spans multiple task types beyond receipts. I also did not use a much larger model because the goal was to stay within the parameter budget while keeping the pipeline reproducible on accessible hardware.

## 2. Why the pipeline is image-first

I chose an image-first pipeline even for PDFs:

1. Many important signals in this assignment are visual rather than purely textual.
2. PDF text extraction would not reliably capture signatures, empty fields, checkboxes, scan artifacts, or rasterized sections.
3. Using the same visual pipeline for PDFs and standalone images keeps the extraction path consistent and easier to evaluate.

That is why `run_pipeline.py` routes both PDFs and images through `pdf_to_image.py`, which renders each page and then applies shared preprocessing before inference.

## 3. Preprocessing decisions

The preprocessing pipeline is intentionally conservative: deskew, denoise, grayscale, and resize with a bounded max side.

The logic here is important. In document extraction, over-processing can destroy exactly the features the model needs:

- aggressive thresholding can erase faint text and signatures
- heavy resizing can distort small receipt text
- strong denoising can remove pen strokes or punctuation

So the design goal was not “clean the image as much as possible.” It was “make the page easier to read while preserving weak but meaningful evidence.” That tradeoff is why the preprocessing in `pdf_to_image.py` is light rather than destructive.

## 4. Prompting decisions

I did not use one generic prompt for every task. Instead, `extract.py` defines:

- one shared system prompt that enforces JSON-only output
- one task-specific prompt per extraction mode
- one expected top-level schema per task
- a repair path when the first output is malformed or schema-incomplete

This was the right decision for two reasons.

First, the tasks have different contracts. Signature detection is a binary visual decision. Form extraction is a structured list problem. Generic document extraction is a broader summarization-plus-entity task. CORD receipt extraction requires a receipt-specific schema with nested totals and menu lines. Using one universal prompt would make the output format less precise and would increase evaluation noise.

Second, the failure modes are separable:

- the model may misunderstand the document
- the model may understand the document but format the answer badly

The repair prompt is designed to address the second problem without changing the overall extraction objective. That is a better engineering pattern than treating every malformed response as a total extraction failure.

I also chose implicit reasoning rather than visible chain-of-thought. The objective here is reliable machine-readable output, not explanation quality. In practice, exposing longer reasoning often increases schema drift and formatting errors. Asking the model to reason internally while forcing a strict JSON contract produced more stable outputs.

## 5. Why the evaluation is split between CORD and generated PDFs

The assignment includes several task categories, but not all of them have a clean public benchmark with the exact required labels. So I split evaluation into two layers:

### Quantitative evaluation on CORD

I used the real CORD dataset for receipt benchmarking because it provides real scanned receipt images with structured annotations. This is the most defensible source for comparing models quantitatively.

`prepare_cord.py` downloads a reproducible subset, converts the original CORD annotations into the project target schema using `cord_utils.py`, and records the selected rows in `metadata.csv`. This was a deliberate decision so the benchmark is both simplified enough for the assignment and still traceable back to a real dataset.

`model_compare.py` then evaluates both models on the same prepared subset. That same-input design matters because it removes sampling noise from the model comparison.

### Qualitative PDF evaluation on generated documents

I used `generate_sample_pdfs.py` to create sample invoices, forms, receipts, and agreement-like documents for a different reason: they exercise the PDF ingestion path and the non-CORD tasks such as signature presence and filled/empty form fields.

I did not treat those PDFs as the main benchmark because they are synthetic and limited in scale. Their job is to demonstrate that the full pipeline works end to end on actual PDF files and task types that CORD does not cover well.

## 6. Why the metrics are structured the way they are

The output is structured JSON, so evaluation should reward field-level correctness rather than only whole-document equality.

That is why `evaluate.py` uses:

- exact match for strict equality
- precision, recall, and F1 over normalized key-value style comparisons

This is a better fit than exact match alone because document extraction often has partial correctness. A model that gets most fields right but misses one total or one menu item should not be scored the same as a totally incorrect extraction. At the same time, exact match is still useful because it measures whether the model can produce a fully correct structured result.

## 7. Fine-tuning decision logic

I added `finetune_lora.py` and `compare_finetuned.py` because the assignment benefits from showing not only zero-shot comparison, but also a realistic adaptation path.

The decision to use LoRA was driven by efficiency:

- it is much lighter than full-model fine-tuning
- it is practical for limited hardware
- it isolates task adaptation into portable adapter weights

I fine-tune the smaller baseline model rather than the stronger primary model because that gives the clearest signal about whether task-specific adaptation can close part of the gap to the better zero-shot model. It is also the more compute-efficient experiment.

`finetune_lora.py` reuses the same core prompt structure as inference and writes training metadata, eval metadata, and the full training configuration. That makes the tuning run reproducible and easier to inspect.

`compare_finetuned.py` exists to answer a very specific technical question: did the adapter improve the model on the same task and same evaluation set, or did it just add complexity? Running base and fine-tuned variants through the same pipeline is the cleanest way to answer that.

## 8. File-by-file implementation logic

- `run_pipeline.py` is the main orchestrator because the project needs one simple entrypoint that can handle PDFs, images, directories, and optional evaluation.
- `pdf_to_image.py` exists as a separate module so rendering and preprocessing stay reusable across CLI, app, comparison, and fine-tuning workflows.
- `extract.py` centralizes model loading, prompt definitions, JSON parsing, and repair so that inference behavior remains consistent everywhere else in the repo.
- `evaluate.py` is isolated so metrics can be reused by the main pipeline, model comparison, and fine-tuning comparison.
- `prepare_cord.py` and `cord_utils.py` separate dataset preparation from modeling logic, which keeps the benchmark reproducible and the extraction code cleaner.
- `model_compare.py` exists because model comparison should be a first-class experiment, not manual shell work.
- `batch_infer.py` exists for the simple case where inputs are already page images and full PDF orchestration would add unnecessary overhead.
- `app.py` exists to make the system easy to demo interactively without changing the core extraction path.

## 9. Limitations

- The CORD benchmark covers receipts well, but not signatures or broader form understanding.
- The synthetic PDF set is useful for demos, but it is not a substitute for a large labeled benchmark.
- Generic VLMs can still hallucinate values when the evidence is weak, which is why the prompts prefer `null` over guessing.
- Fine-tuning here is a starter adaptation path, not a fully scaled training study.

## 10. Production improvements

- Add document-type routing so receipts, forms, and agreements use specialized prompts automatically.
- Add confidence estimation and human-review fallback for low-confidence fields.
- Expand evaluation with a labeled signature/form dataset instead of synthetic-only checks.
- Increase the CORD evaluation size and run training / inference on stronger GPU hardware.
- Version prompts, configs, and outputs more formally for auditability.
