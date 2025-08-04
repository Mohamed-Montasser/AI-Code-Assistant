import gradio as gr
from typing import Optional
# #
# !pip install google-generativeai
# !pip install requests
# !pip install datasets sentence-transformers faiss-cpu transformers torch
import os
import google.generativeai as genai
import numpy as np
from typing import List
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Tuple
import re
import json
import requests
import json
import re
from typing import List, Dict
import google.generativeai as genai  # Add this import
import time  # Add this import at the top
from requests.exceptions import HTTPError
import requests
import re
from typing import List, Dict

class HumanEvalDatasetProcessor:
    """Handles loading and processing of HumanEval dataset"""
    def __init__(self):
        self.dataset = None
        self.processed_data = None

    def load_dataset(self):
        """Load HumanEval dataset from Hugging Face"""
        print("Loading HumanEval dataset...")
        self.dataset = load_dataset("openai_humaneval", split="test")
        print(f"Loaded {len(self.dataset)} examples")
        return self.dataset

    def extract_fields(self):
        """Extract task_id, prompt, and canonical_solution fields"""
        if self.dataset is None:
            self.load_dataset()

        self.processed_data = []
        for example in self.dataset:
            processed_example = {
                'task_id': example['task_id'],
                'prompt': example['prompt'],
                'canonical_solution': example['canonical_solution'],
                'entry_point': example['entry_point']
            }
            self.processed_data.append(processed_example)

        print(f"Processed {len(self.processed_data)} examples")
        return self.processed_data

    def get_dataframe(self):
        """Convert processed data to pandas DataFrame"""
        if self.processed_data is None:
            self.extract_fields()

        return pd.DataFrame(self.processed_data)

class EmbeddingPipeline:
    """Handles text embedding using Gemini's embedding API"""

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.embeddings = None

        if api_key:
            genai.configure(api_key=api_key)

    def load_model(self):
        """Verify API configuration"""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        print(f"Using Gemini embedding model: {self.model_name}")

    def embed_prompts(self, prompts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple prompts"""
        if not hasattr(self, 'api_key'):
            self.load_model()

        print(f"Generating embeddings for {len(prompts)} prompts...")
        embeddings = []
        for prompt in prompts:
            result = genai.embed_content(
                model=self.model_name,
                content=prompt,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

        self.embeddings = np.array(embeddings)
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    def embed_single_prompt(self, prompt: str) -> np.ndarray:
        """Generate embedding for a single prompt"""
        return self.embed_prompts([prompt])[0]

class VectorDatabase:
    """Manages vector storage and retrieval using FAISS"""

    def __init__(self, dimension: int = 768):  # Gemini embeddings are 768-dimensional
        self.dimension = dimension
        self.index = None
        self.data = None

    def create_index(self, embeddings: np.ndarray, data: List[Dict]):
        """Create FAISS index from embeddings"""
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        self.data = data
        print(f"Index created with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for top-k similar examples"""
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")

        # Ensure query_embedding has the right shape (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Perform similarity search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        # Return top-k results
        results = []
        for i in range(min(k, len(indices[0]))):  # Handle case where k > available vectors
            idx = indices[0][i]
            if idx < len(self.data):  # Ensure index is valid
                result = self.data[idx].copy()
                result['similarity_score'] = float(distances[0][i])
                result['rank'] = i + 1
                results.append(result)

        return results

class RetrievalEngine:
    """Combines embedding and vector database for retrieval"""

    def __init__(self, embedding_model_name: str = None, embedding_pipeline=None, api_key: str = None):
        if embedding_pipeline is None:
            if embedding_model_name is None:
                embedding_model_name = "models/embedding-001"
            self.embedding_pipeline = EmbeddingPipeline(embedding_model_name, api_key)
        else:
            self.embedding_pipeline = embedding_pipeline

        # Get dimension from the first embedding (Gemini returns 768-dim vectors)
        self.vector_db = VectorDatabase(dimension=768)

    def build_index(self, data: List[Dict]):
        """Build the retrieval index from dataset"""
        # Extract prompts for embedding
        prompts = [item['prompt'] for item in data]

        # Generate embeddings
        embeddings = self.embedding_pipeline.embed_prompts(prompts)

        # Create vector database index
        self.vector_db.create_index(embeddings, data)

        print("Retrieval index built successfully")

    def retrieve_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k similar examples for a query"""
        # Generate embedding for query
        query_embedding = self.embedding_pipeline.embed_single_prompt(query)

        # Search in vector database
        results = self.vector_db.search(query_embedding, k)

        return results

class CodeGenerator:
    """Generates code using Gemini API with retrieved context"""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.configured = False

    def load_model(self):
        """Configure the Gemini API client"""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        if not self.configured:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.configured = True
            print(f"Gemini model {self.model_name} configured successfully")

    def generate_code(self, user_prompt: str, retrieved_examples: List[Dict]) -> str:
        """Generate code with robust error handling"""
        self.load_model()  # Ensure configuration

        payload = self.create_context_prompt(user_prompt, retrieved_examples)
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.api_key
        }

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                response_data = response.json()
                return self.extract_code(response_data['candidates'][0]['content']['parts'][0]['text'])

            except HTTPError as e:
                last_exception = e
                if e.response.status_code in [503, 429]:  # Service Unavailable or Too Many Requests
                    print(f"Attempt {attempt + 1} failed with status {e.response.status_code}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise ValueError(f"API request failed: {str(e)}")

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
                    time.sleep(self.retry_delay)
                    continue
                raise ValueError(f"API request failed after {self.max_retries} attempts: {str(e)}")

        raise ValueError(f"API request failed after {self.max_retries} attempts: {str(last_exception)}")

    def create_context_prompt(self, user_prompt: str, retrieved_examples: List[Dict]) -> Dict:
        """Create the payload for Gemini API"""
        context = "You are an expert Python programmer. Here are some similar coding tasks and their solutions:\n\n"

        for i, example in enumerate(retrieved_examples, 1):
            context += f"Example {i} (Task ID: {example['task_id']}):\n"
            context += f"Task Description:\n{example['prompt']}\n"
            context += f"Solution:\n```python\n{example['canonical_solution']}\n```\n\n"

        context += f"Now, please solve this new task:\n{user_prompt}\n\n"
        context += "Provide only the Python code solution in a code block, with no additional explanation."

        return {
            "contents": [{
                "parts": [{
                    "text": context
                }]
            }]
        }

    def extract_code(self, generated_text: str) -> str:
        """Extract code from response"""
        code_blocks = re.findall(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else generated_text.strip()


class IntentClassifier:
    """Classifies user intent using Gemini LLM"""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.configured = False

    def load_model(self):
        """Configure the Gemini API client"""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        if not self.configured:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.configured = True
            print(f"Gemini model {self.model_name} configured for intent classification")

    def classify_intent(self, user_input: str) -> Dict:
        """Classify user intent and return structured JSON"""
        self.load_model()  # Ensure configuration

        prompt = f"""
        Analyze the following user input and classify its intent for a coding assistant system.
        Return ONLY a JSON response with the structure: {{"task": "task_type", "user_input": "original_input"}}

        Possible task types:
        - "code_generation": User wants to generate new code
        - "code_explanation": User wants an explanation of code
        - "code_debugging": User wants help debugging code
        - "code_optimization": User wants to optimize existing code
        - "unknown": Cannot determine intent

        User Input: {user_input}
        """

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)

            # Extract JSON from response
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]  # Remove markdown json tags if present

            result = json.loads(json_str)
            return {
                "task": result.get("task", "unknown"),
                "user_input": result.get("user_input", user_input)
            }
        except Exception as e:
            print(f"Intent classification failed: {str(e)}")
            return {"task": "unknown", "user_input": user_input}

class RAGCodeGenerator:
    """Main pipeline that integrates all components with smart routing"""

    def __init__(self,
                 embedding_model: str = "models/embedding-001",
                 code_model: str = "gemini-1.5-flash",
                 gemini_api_key: str = None):
        """
        Initialize the RAG pipeline with intent classification
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        self.dataset_processor = HumanEvalDatasetProcessor()
        self.retrieval_engine = RetrievalEngine(
            embedding_model_name=embedding_model,
            api_key=gemini_api_key
        )
        self.code_generator = CodeGenerator(
            model_name=code_model,
            api_key=gemini_api_key
        )
        self.intent_classifier = IntentClassifier(
            model_name=code_model,  # Can use same model
            api_key=gemini_api_key
        )
        self.ready = False

    def setup(self):
        """Initialize the complete pipeline"""
        print("Setting up RAG Code Generation Pipeline with Intent Routing...")
        processed_data = self.dataset_processor.extract_fields()
        self.retrieval_engine.build_index(processed_data)
        self.ready = True
        print("Pipeline setup complete!")

    def route_intent(self, user_input: str) -> Dict:
        """Classify user intent and route to appropriate handler"""
        intent = self.intent_classifier.classify_intent(user_input)
        print(f"Intent classified as: {intent['task']}")
        return intent

    def generate(self, user_input: str, k: int = 3) -> Dict:
        """Generate response based on classified intent"""
        if not self.ready:
            raise ValueError("Pipeline not setup. Call setup() first.")

        # First classify intent
        intent = self.route_intent(user_input)

        # Route to appropriate handler
        if intent["task"] == "code_generation":
            return self.handle_code_generation(intent["user_input"], k)
        elif intent["task"] == "code_explanation":
            return self.handle_code_explanation(intent["user_input"])
        elif intent["task"] == "code_debugging":
            return self.handle_code_debugging(intent["user_input"])
        elif intent["task"] == "code_optimization":
            return self.handle_code_optimization(intent["user_input"])
        else:
            return self.handle_unknown_intent(intent["user_input"])

    def handle_code_generation(self, user_prompt: str, k: int) -> Dict:
        """Handle code generation request"""
        print(f"Processing code generation request: {user_prompt[:100]}...")
        retrieved_examples = self.retrieval_engine.retrieve_similar(user_prompt, k)
        generated_code = self.code_generator.generate_code(user_prompt, retrieved_examples)

        return {
            'task': 'code_generation',
            'user_input': user_prompt,
            'retrieved_examples': retrieved_examples,
            'generated_code': generated_code
        }

    def handle_code_explanation(self, user_input: str) -> Dict:
        """Handle code explanation request"""
        print(f"Processing code explanation request: {user_input[:100]}...")
        # Implementation would use LLM to explain code
        return {
            'task': 'code_explanation',
            'user_input': user_input,
            'explanation': "Explanation would go here"
        }

    def handle_code_debugging(self, user_input: str) -> Dict:
        """Handle code debugging request"""
        print(f"Processing code debugging request: {user_input[:100]}...")
        # Implementation would use LLM to debug code
        return {
            'task': 'code_debugging',
            'user_input': user_input,
            'debugging_suggestions': "Debugging suggestions would go here"
        }

    def handle_code_optimization(self, user_input: str) -> Dict:
        """Handle code optimization request"""
        print(f"Processing code optimization request: {user_input[:100]}...")
        # Implementation would use LLM to optimize code
        return {
            'task': 'code_optimization',
            'user_input': user_input,
            'optimized_code': "Optimized code would go here"
        }

    def handle_unknown_intent(self, user_input: str) -> Dict:
        """Handle unknown intent"""
        return {
            'task': 'unknown',
            'user_input': user_input,
            'message': "I'm not sure what you're asking for. Could you clarify?"
        }

    def generate_response(self, user_input: str, k: int = 3) -> None:
        """Generate and print response in the requested format"""
        # Classify intent
        intent = self.route_intent(user_input)

        # Generate response based on intent
        if intent["task"] == "code_generation":
            result = self.handle_code_generation(intent["user_input"], k)
            print(f"Task type: {intent['task']}")
            print(f"User input: {result['user_input']}")
            print("Generated code:")
            print(result['generated_code'])
        else:
            # Handle other task types
            print(f"Task type: {intent['task']}")
            print(f"User input: {intent['user_input']}")
            print("Response:")
            print(self.handle_non_generation_task(intent))

    def handle_non_generation_task(self, intent: dict) -> str:
        """Handle non-code-generation tasks"""
        if intent["task"] == "code_explanation":
            return "Code explanation would go here"
        elif intent["task"] == "code_debugging":
            return "Debugging suggestions would go here"
        elif intent["task"] == "code_optimization":
            return "Optimized code would go here"
        else:
            return "I'm not sure what you're asking for. Could you clarify?"

# Initialize the pipeline
api_key = os.getenv("geminiAPI")
rag_generator = RAGCodeGenerator(gemini_api_key = api_key)
rag_generator.setup()

# Store conversation history
conversation_history = []


def chat_with_ai(user_input: str, history: list) -> Tuple[str, list]:
    """Handle continuous chat until 'quit' is entered"""
    global conversation_history

    if user_input.lower() == 'quit':
        conversation_history = []  # Reset for next session
        return "Session ended. Refresh the page to start a new session.", []

    try:
        # Generate the response
        result = rag_generator.generate(user_input, k=3)

        # Format the response
        response = f"Task type: code_generation\n"
        response += f"User input: {result['user_input']}\n\n"
        response += "Generated code:\n"
        response += result['generated_code']

        # Update history
        history.append((user_input, response))
        conversation_history = history

        return "", history  # Empty string for new input, history for display

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((user_input, error_msg))
        conversation_history = history
        return "", history


# Create Gradio interface with chat UI
with gr.Blocks(title="AI Code Generator Chat") as demo:
    gr.Markdown("## AI Code Generator Chat")
    gr.Markdown("Enter coding tasks one at a time. Type 'quit' to end the session.")

    chatbot = gr.Chatbot(label="Conversation History")
    msg = gr.Textbox(label="Enter your coding task", placeholder="Type your request here...")
    clear = gr.Button("Clear Conversation")


    def respond(message, chat_history):
        return chat_with_ai(message, chat_history)


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
