"""LangSmith evaluation and tracing configuration."""

import os
from typing import Any, Callable
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


class LangSmithEvaluator:
    """LangSmith evaluation service for RAG and weather responses."""
    
    def __init__(
        self,
        api_key: str | None = None,
        project_name: str = "weather-rag-pipeline",
    ):
        """Initialize LangSmith evaluator.
        
        Args:
            api_key: LangSmith API key.
            project_name: Project name for tracing.
        """
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
        self.project_name = project_name
        
        # Set environment variables for LangSmith
        if self.api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = project_name
        
        self._client = None
    
    @property
    def client(self) -> Client:
        """Get or create LangSmith client."""
        if self._client is None and self.api_key:
            self._client = Client(api_key=self.api_key)
        return self._client
    
    @property
    def is_enabled(self) -> bool:
        """Check if LangSmith is enabled."""
        return bool(self.api_key)
    
    def configure_tracing(self) -> dict:
        """Configure tracing environment variables.
        
        Returns:
            Dictionary of configured environment variables.
        """
        config = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_PROJECT": self.project_name,
        }
        
        if self.api_key:
            config["LANGCHAIN_API_KEY"] = self.api_key
        
        for key, value in config.items():
            os.environ[key] = value
        
        return config
    
    def create_relevance_evaluator(self, llm: ChatOpenAI) -> Callable:
        """Create a custom relevance evaluator.
        
        Args:
            llm: Language model for evaluation.
            
        Returns:
            Evaluator function.
        """
        def evaluate_relevance(run, example) -> dict:
            """Evaluate if the response is relevant to the query."""
            query = run.inputs.get("query", "")
            response = run.outputs.get("response", "")
            
            prompt = f"""Rate the relevance of this response to the query on a scale of 1-10.

Query: {query}
Response: {response}

Consider:
1. Does the response directly address the query?
2. Is the information provided accurate and helpful?
3. Is there unnecessary or off-topic information?

Respond with ONLY a number from 1-10."""

            result = llm.invoke([HumanMessage(content=prompt)])
            
            try:
                score = int(result.content.strip()) / 10
            except ValueError:
                score = 0.5
            
            return {
                "key": "relevance",
                "score": score,
            }
        
        return evaluate_relevance
    
    def create_faithfulness_evaluator(self, llm: ChatOpenAI) -> Callable:
        """Create a custom faithfulness evaluator for RAG responses.
        
        Args:
            llm: Language model for evaluation.
            
        Returns:
            Evaluator function.
        """
        def evaluate_faithfulness(run, example) -> dict:
            """Evaluate if the response is faithful to the context."""
            context = run.outputs.get("context", "")
            response = run.outputs.get("response", "")
            
            if not context:
                return {"key": "faithfulness", "score": 1.0}
            
            prompt = f"""Evaluate if the response is faithful to the provided context.
Rate on a scale of 1-10 where:
- 10 = Response is completely based on the context
- 1 = Response contains information not in the context

Context: {context[:2000]}

Response: {response}

Respond with ONLY a number from 1-10."""

            result = llm.invoke([HumanMessage(content=prompt)])
            
            try:
                score = int(result.content.strip()) / 10
            except ValueError:
                score = 0.5
            
            return {
                "key": "faithfulness",
                "score": score,
            }
        
        return evaluate_faithfulness
    
    def create_coherence_evaluator(self, llm: ChatOpenAI) -> Callable:
        """Create a coherence evaluator.
        
        Args:
            llm: Language model for evaluation.
            
        Returns:
            Evaluator function.
        """
        def evaluate_coherence(run, example) -> dict:
            """Evaluate if the response is coherent and well-structured."""
            response = run.outputs.get("response", "")
            
            prompt = f"""Rate the coherence of this response on a scale of 1-10.

Response: {response}

Consider:
1. Is the response well-organized?
2. Does it flow logically?
3. Is it easy to understand?

Respond with ONLY a number from 1-10."""

            result = llm.invoke([HumanMessage(content=prompt)])
            
            try:
                score = int(result.content.strip()) / 10
            except ValueError:
                score = 0.5
            
            return {
                "key": "coherence",
                "score": score,
            }
        
        return evaluate_coherence
    
    def evaluate_response(
        self,
        llm: ChatOpenAI,
        query: str,
        response: str,
        context: str = "",
        query_type: str = "document",
    ) -> dict[str, float]:
        """Evaluate a single response.
        
        Args:
            llm: Language model for evaluation.
            query: Original query.
            response: Generated response.
            context: Context used for generation (for RAG).
            query_type: Type of query (weather/document).
            
        Returns:
            Dictionary of evaluation scores.
        """
        scores = {}
        
        # Relevance evaluation
        relevance_prompt = f"""Rate the relevance of this response to the query on a scale of 1-10.

Query: {query}
Response: {response}

Respond with ONLY a number from 1-10."""

        relevance_result = llm.invoke([HumanMessage(content=relevance_prompt)])
        try:
            scores["relevance"] = int(relevance_result.content.strip()) / 10
        except ValueError:
            scores["relevance"] = 0.5
        
        # Coherence evaluation
        coherence_prompt = f"""Rate the coherence and clarity of this response on a scale of 1-10.

Response: {response}

Respond with ONLY a number from 1-10."""

        coherence_result = llm.invoke([HumanMessage(content=coherence_prompt)])
        try:
            scores["coherence"] = int(coherence_result.content.strip()) / 10
        except ValueError:
            scores["coherence"] = 0.5
        
        # Faithfulness evaluation (for RAG responses only)
        if query_type == "document" and context:
            faithfulness_prompt = f"""Evaluate if the response is faithful to the context (1-10).
10 = completely based on context, 1 = contains hallucinations.

Context: {context[:2000]}
Response: {response}

Respond with ONLY a number from 1-10."""

            faithfulness_result = llm.invoke([HumanMessage(content=faithfulness_prompt)])
            try:
                scores["faithfulness"] = int(faithfulness_result.content.strip()) / 10
            except ValueError:
                scores["faithfulness"] = 0.5
        
        # Calculate overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def log_evaluation(
        self,
        query: str,
        response: str,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log evaluation results to LangSmith.
        
        Args:
            query: Original query.
            response: Generated response.
            scores: Evaluation scores.
            metadata: Additional metadata to log.
        """
        if not self.is_enabled:
            return
        
        # Evaluation results are automatically logged through tracing
        # This method is for additional custom logging if needed
        print(f"Evaluation logged - Overall score: {scores.get('overall', 0):.2f}")
    
    def get_project_stats(self) -> dict[str, Any] | None:
        """Get statistics for the current project.
        
        Returns:
            Project statistics or None if not available.
        """
        if not self.client:
            return None
        
        try:
            # Get recent runs
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=100,
            ))
            
            return {
                "project_name": self.project_name,
                "total_runs": len(runs),
                "recent_runs": [
                    {
                        "id": str(run.id),
                        "name": run.name,
                        "status": run.status,
                        "start_time": str(run.start_time) if run.start_time else None,
                    }
                    for run in runs[:10]
                ],
            }
        except Exception as e:
            return {"error": str(e)}


def setup_langsmith_tracing(
    api_key: str | None = None,
    project_name: str = "weather-rag-pipeline",
) -> LangSmithEvaluator:
    """Setup LangSmith tracing for the application.
    
    Args:
        api_key: LangSmith API key.
        project_name: Project name for tracing.
        
    Returns:
        Configured LangSmithEvaluator.
    """
    evaluator = LangSmithEvaluator(
        api_key=api_key,
        project_name=project_name,
    )
    evaluator.configure_tracing()
    return evaluator
