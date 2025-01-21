Intuitive Physical Reasoning with LLM: Balancing Computational Cost and Performance

Authors

Vidhi Bhatt, Gary Huang, Yuxuan Xia

University of Michigan

Project Overview

This project explores the performance of two state-of-the-art NLP models, Llama-3 and BitNet, on the Tiered Reasoning for Intuitive Physics (TRIP) dataset. We assess their ability to understand and reason about physical phenomena using prompting techniques. The study highlights the trade-offs between computational efficiency and reasoning performance, contributing to the advancement of NLP methodologies for intuitive physics tasks.

Research Objectives

	•	Evaluate Llama-3 and BitNet on the TRIP dataset.
	•	Compare reasoning abilities, accuracy, and computational efficiency.
	•	Explore prompting techniques (zero-shot, few-shot, chain-of-thought) to enhance model performance.
	•	Investigate trade-offs between large-scale transformers and lightweight quantized models.

Dataset: Tiered Reasoning for Intuitive Physics (TRIP)

The TRIP dataset is designed to test commonsense reasoning in physics through multi-tiered benchmarks:

	•	Story Plausibility Identification – Determine which story is logically plausible.
	•	Conflict Sentence Detection – Identify inconsistent sentences in implausible stories.
	•	Physical State Attribution – Recognize physical principles behind implausibility.

Models Used

	1.	Meta-Llama-3-8B-Instruct
	•	Transformer-based model optimized for general NLP tasks and reasoning.
	•	Fine-tuned for instruction-following tasks.
	2.	HF1BitLLM/Llama3-8B-1.58-100B-tokens (BitNet)
	•	1-bit quantized model for computational efficiency.
	•	Designed to perform well in resource-constrained environments.

Experimental Approach

	•	Baseline Prompting: Standard input-output without additional reasoning guidance.
	•	Reasoning Prompting: Chain-of-thought prompting to guide models through reasoning steps.
	•	One-shot Prompting: Provide a single high-quality example to improve model responses.

Implementation Details

	•	Models were evaluated using llama.cpp, optimized for CPU inference.
	•	Experiments were run on an AMD Ryzen 9 3900x CPU using single-core processing.
	•	Fine-tuning was attempted but not pursued due to llama.cpp limitations.
	•	Used Backus-Naur Form (BNF) constraints to format structured outputs.

Results Summary

Model	Approach	Story Accuracy	Sentence Accuracy	Avg. Inference Time (s)

Llama-3	Baseline	55.0%	77.0%	116.11

BitNet	Baseline	49.0%	41.0%	51.28

Llama-3	Reasoning	56.0%	66.0%	264.22

BitNet	Reasoning	51.0%	30.0%	119.78

Llama-3	One-shot	59.0%	62.0%	329.99

BitNet	One-shot	46.0%	34.0%	159.03

	•	Llama-3 consistently performed better in accuracy but had higher inference time.
	•	BitNet was significantly faster but struggled with lower reasoning accuracy.
	•	Reasoning and One-shot prompting improved performance for Llama but had little impact on BitNet.

Key Takeaways

	•	Llama-3 is better for reasoning-intensive tasks, but BitNet is more efficient.
	•	Prompting strategies improve accuracy, but their impact depends on the model.
	•	1-bit quantization reduces computational cost, but it needs further optimization for reasoning tasks.
	•	Models struggle with maintaining temporal order, requiring better structured prompting.

Future Directions

	•	Improve quantization methods to enhance lightweight model performance.
	•	Experiment with multi-modal datasets combining text and vision for better physics-based reasoning.
	•	Explore fine-tuning optimizations to better align model predictions with human commonsense.

How to Run
	1.	Clone the repository:

git clone https://github.com/your-repo-name.git

cd your-repo-name


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Run inference using llama.cpp:

python run_experiment.py --model llama3 --dataset trip



Contributors
	•	Vidhi Bhatt: Report writing, fine-tuning attempts, methodology documentation.
	•	Gary Huang: Technical setup, llama.cpp integration, running experiments.
	•	Yuxuan Xia: Literature review, presentations, prompt optimization.

References
	•	TRIP Dataset Paper
	•	BitNet Paper
	•	Llama.cpp GitHub
	•	Llama-3 Model Card

This README provides a structured and professional overview of your project, making it easier for others to understand and replicate your work. Let me know if you need any modifications! 🚀
