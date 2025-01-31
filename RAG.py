import asyncio
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    generate_question_context_pairs
)


class RAGPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, dataset_dir, chunk_size, top_k, model_name, embedder_name, dataset_name, retriever_type):
        """
        Initialize the RAGPredictor class with necessary components.
        """
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.model_name = model_name
        self.embedder_name = embedder_name
        self.dataset_name = dataset_name
        self.retriever_type = retriever_type
        

        from llama_index.core import Settings

        # Suppression de OpenAI, et utilisation de Ollama uniquement pour les embeddings
        self.embed_model = OllamaEmbedding(
                    model_name="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
        )

        Settings.embed_model = self.embed_model


        self.llm = LlamaIndexOllama(
                    model="llama3.2:latest",
                    base_url="http://localhost:11434"
        )
        Settings.llm = self.llm

        if not self.llm or not self.embed_model:
          raise ValueError("LLM or Embedding model initialization failed. Ensure Ollama is configured correctly.")

        # Load and process documents
        self.nodes = self._prepare_nodes()

        # Initialize retrievers
        self.vector_index = VectorStoreIndex(self.nodes)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k, similarity_cutoff=0.6)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes)

    def __getstate__(self):
        state = self.__dict__.copy()
        exclude_keys = ["vector_index", "vector_retriever", "bm25_retriever", "nodes"]  # Ajustez la liste
        for key in exclude_keys:
            state.pop(key, None)
        return state



    def __setstate__(self, state):
        self.__dict__.update(state)
        # Réinitialisation explicite des modèles Ollama
        self.llm = LlamaIndexOllama(
            model=self.model_name,
            base_url="http://localhost:11434"
        )
        self.embed_model = OllamaEmbedding(
            model_name=self.embedder_name,
            base_url="http://localhost:11434"
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Préparation des nodes et retrievers
        self.nodes = self._prepare_nodes()
        self.vector_index = VectorStoreIndex(self.nodes)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k, similarity_cutoff=0.6)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes)

        # Vérifiez que le retriever est bien initialisé
        if self.vector_retriever is None:
           raise ValueError("vector_retriever is not initialized properly")

    def _prepare_nodes(self):
        """
        Load documents and create nodes using SentenceSplitter.
        """
        documents = SimpleDirectoryReader(self.dataset_dir).load_data()
        node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents(documents)
        for idx, node in enumerate(nodes):
            node.id_ = f"node_{idx}"
        return nodes

    def predict(self, context, input_data):
        """
        MLflow expects this method to make predictions.
            """
        # Ensure global settings are updated
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        llm_model = LlamaIndexOllama(
            model="llama3.2:latest",
            base_url="http://localhost:11434"
        )
        Settings.embed_model = embed_model
        Settings.llm = llm_model

        # Ensure input_data is a dictionary with a 'query' key
        if isinstance(input_data, pd.DataFrame):
            query = input_data['question'].iloc[0]
        elif isinstance(input_data, dict):
            query = input_data.get('question', '')
        else:
            raise ValueError("Unsupported input format")
        
            # Préparation des nodes et retrievers
        self.nodes = self._prepare_nodes()
        self.vector_index = VectorStoreIndex(self.nodes, embed_model=self.embed_model)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k, similarity_cutoff=0.6) 
        # Retrieve relevant nodes
        results = self.vector_retriever.retrieve(query[0])

        # Combine results into a response
        response = "\n".join([node.text for node in results])
        return response

    async def evaluate(self):
        """
        Evaluate the retriever on the dataset and log metrics to MLflow.
        """
        # Check if the dataset JSON file exists
        if os.path.exists(self.dataset_name):
            # Load existing dataset
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(self.dataset_name)
            print(f"Loaded existing dataset from {self.dataset_name}")
        else:
            # Generate new dataset
            qa_dataset = generate_question_context_pairs(
                self.nodes,
                llm=self.llm,
                num_questions_per_chunk=self.chunk_questions
            )
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset)
            # Save the newly generated dataset
            qa_dataset.save_json(self.dataset_name)
            print(f"Generated and saved new dataset to {self.dataset_name}")

        # Define metrics and evaluator
        metrics = ["mrr", "hit_rate"]
        retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=self.vector_retriever)

        # Evaluate the dataset
        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        # Calculate averages
        full_df = pd.DataFrame(metric_dicts)
        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_metric("Hit Rate", hit_rate)
            mlflow.log_metric("MRR", mrr)

            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("embedder_name", self.embedder_name)
            mlflow.log_param("top_k", self.top_k)
            mlflow.log_param("chunk_size", self.chunk_size)
            mlflow.log_param("retriever type", self.retriever_type)
            mlflow.log_artifact(self.dataset_name)

        print("Evaluation completed. View your results at http://127.0.0.1:5000")

    def log_and_register_model(self):
        """
        Log the model into MLflow and register it for deployment.
        """
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("embedder_name", self.embedder_name)
            mlflow.log_param("top_k", self.top_k)
            mlflow.log_param("chunk_size", self.chunk_size)
            mlflow.log_param("retriever type", self.retriever_type)

            # Log the model
            mlflow.pyfunc.log_model(
                artifact_path="rag",
                python_model=predictor,
                # registered_model_name="RAGPredictorModel"
        )

        print("Model logged and registered in MLflow.")

# Example usage
if __name__ == "__main__":
    predictor = RAGPredictor(
        dataset_dir="./data",
        chunk_size=512,
        top_k=5,
        model_name="llama3.2:latest",
        embedder_name="nomic-embed-text:latest",
        dataset_name="lois.json",
        retriever_type="vector_retriever"
    )

    async def main():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("rag")
        predictor.log_and_register_model()

        # Tester la méthode predict avec une question
        try:
            question = {"question": ["What is the date of promulgation of the law according to the given text?"]}
            response = predictor.predict(None, question)
            print("Response from predict():", response)
        except Exception as e:
            print("An error occurred during prediction:", e)

    asyncio.run(main())
