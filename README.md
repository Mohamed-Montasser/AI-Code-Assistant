# 🤖 AI Code Assistant with Gemini RAG

A smart code generation assistant that uses **Google Gemini LLMs**, **retrieval-augmented generation (RAG)**, and **intent classification** to provide contextual, high-quality code generation from user prompts. Built with Gradio for interactive chat.

---

### 🌐 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/M-Montasser/RagCodeGenerator)

Try the app instantly in your browser — no installation needed!

---

## 🚀 Features

- 🔍 **Intent Classification**: Automatically detects user intent (e.g., generate, explain, debug, optimize code).
- 📚 **Retrieval-Augmented Generation**: Uses HumanEval dataset with Gemini embeddings and FAISS vector search to provide similar code examples.
- 🧠 **Code Generation**: Gemini LLM generates Python code conditioned on user prompts and retrieved examples.
- 🎯 **Context-Aware Responses**: Responses are tailored with relevant examples to enhance code quality.
- 💬 **Interactive Chat Interface**: Gradio-powered interface for natural back-and-forth with the assistant.

---

## 🛠️ Technologies Used

- `Gradio` – for UI interface.
- `Google Generative AI (Gemini)` – for both text generation and embedding.
- `FAISS` – for fast vector search and retrieval.
- `Hugging Face Datasets` – HumanEval coding benchmark.
- `Sentence-Transformers` – optional text embedding support.
- `Transformers / PyTorch` – optional language model interfacing.
- `NumPy / Pandas` – for data processing.

---

## 🧪 How It Works

1. **Dataset Preparation**: Loads the HumanEval dataset and extracts task prompts and solutions.
2. **Embedding Generation**: Converts prompts into vector embeddings using Gemini’s `embedding-001` model.
3. **Vector Indexing**: Uses FAISS to index those embeddings for efficient nearest neighbor retrieval.
4. **Intent Classification**: Classifies the user’s intent to route to the proper action (e.g., generate code).
5. **Code Generation**: Constructs a prompt with retrieved examples and sends it to Gemini for code generation.
6. **Gradio Chat**: Provides a user-friendly interface for interaction.

---

## 💻 Usage

### Run Locally

```bash
python your_script_name.py
```

It will start a Gradio web interface. You can optionally launch it publicly using:

```python
demo.launch(share=True)
```

---

## 🧠 Supported Intents

| Intent Type         | Behavior                            |
|---------------------|-------------------------------------|
| `code_generation`   | Generates Python code from prompt   |
| `code_explanation`  | Explains code (stub implementation) |
| `code_debugging`    | Suggests debugging help (stub)      |
| `code_optimization` | Optimizes existing code (stub)      |
| `unknown`           | Asks user for clarification         |

> Currently, only `code_generation` is fully implemented. Other tasks are placeholders for future enhancements.

---

## 📁 Project Structure

```
.
├── main.py                    # Main Gradio app with pipeline
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
```

---

## ✨ Example Prompt

**User Input**:
```
Write a Python function that finds the maximum sum of a contiguous subarray.
```

**Generated Output**:
```python
def max_subarray_sum(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

---

## 📌 Future Enhancements

- Implement full support for explanation, debugging, and optimization.
- Add session logging and export features.
- Extend dataset support for more task types.
- Enable LangGraph-based workflow routing (optional).

---

## 👌 Acknowledgements

- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [OpenAI HumanEval Dataset](https://huggingface.co/datasets/openai_humaneval)
- [Gradio](https://gradio.app/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📄 License

This project is licensed under the MIT License.

