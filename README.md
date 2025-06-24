#en-hi-llm-finetuning

Fine-tuning Lightweight Transformer Models for English to Hindi Translation using the IIT Bombay Parallel Corpus.

⸻

📌 Overview

This project demonstrates the fine-tuning of pretrained transformer models for English-to-Hindi machine translation, with a focus on efficiency and performance. The core of the work involves adapting the MarianMT (Helsinki-NLP/opus-mt-en-hi) model to the IIT Bombay English-Hindi Parallel Corpus, using techniques like LoRA (Low-Rank Adaptation) to make training feasible on resource-constrained hardware.

⸻

🗂️ Dataset
	•	IIT Bombay English-Hindi Parallel Corpus
	•	Contains ~1.6 million high-quality sentence pairs.
	•	Available here.
	•	Preprocessed using:
	•	Tokenization
	•	Sequence truncation (for maximum model input length)

⸻

🧠 Models & Techniques

✅ Fine-tuned Base Model
	•	Helsinki-NLP/opus-mt-en-hi (MarianMT)
Pretrained model for English to Hindi machine translation, fine-tuned on the IITB dataset.

⚙️ Parameter-Efficient Fine-tuning
	•	LoRA (Low-Rank Adaptation)
	•	Reduced trainable parameters by 99.2% (from 75.8M → 0.59M)
	•	Enabled efficient training on low-resource hardware setups

🧪 Optimization
	•	Optimizer: AdamW
	•	Learning Rate Scheduler: Linear decay with warmup
	•	Evaluation Metrics:
	•	BLEU
	•	METEOR
	•	ROUGE

⸻

📈 Results

Significant performance improvements were observed after fine-tuning:

Metric	Before	After	Improvement
METEOR	0.27	0.32	+18.8%
BLEU	9.8	12.0	+22.7%


⸻

🚀 Quickstart

1. Clone the Repository

git clone https://github.com/your-username/en-hi-llm-finetuning.git
cd en-hi-llm-finetuning

2. Install Dependencies

pip install -r requirements.txt

3. Preprocess Dataset

Prepare and tokenize the IITB dataset.

python scripts/preprocess.py --data_dir data/iitb --tokenizer helsinki

4. Fine-tune the Model

python train.py --model helsinki --use_lora --output_dir checkpoints/opus-mt-en-hi-lora

5. Evaluate

python evaluate.py --model checkpoints/opus-mt-en-hi-lora

-----------

📄 License

This project is licensed under the MIT License.
Note: The IIT Bombay dataset is released under a separate license — please refer to their official terms.

⸻

🙌 Acknowledgements
	•	IIT Bombay English-Hindi Parallel Corpus
	•	Helsinki-NLP / MarianMT models
	•	LoRA: Low-Rank Adaptation of Large Language Models
	•	Hugging Face Transformers
