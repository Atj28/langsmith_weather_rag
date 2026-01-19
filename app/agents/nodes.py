"""LangGraph node implementations."""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from app.services.weather import WeatherService, WeatherServiceError
from app.rag.retriever import RAGRetriever


# Router prompts
ROUTER_SYSTEM_PROMPT = """You are a query classifier. Your job is to determine whether a user's query is about:
1. WEATHER - Current weather conditions, forecasts, temperature, humidity, wind, etc. for any location
2. DOCUMENT - Questions about information in documents, weather forecasting methods, climate data, or any other document-based queries
3. GREETING - Simple greetings, hellos, or casual conversation that doesn't need weather data or documents

Respond with ONLY one word: either "weather", "document", or "greeting".

Examples:
- "What's the weather in New York?" -> weather
- "Will it rain tomorrow in London?" -> weather
- "What is climate normal?" -> document
- "How do weather forecasts work?" -> document
- "Temperature in Paris" -> weather
- "Explain weather prediction models" -> document
- "hello" -> greeting
- "hi" -> greeting
- "hey there" -> greeting
- "how are you?" -> greeting
- "good morning" -> greeting
"""

WEATHER_RESPONSE_PROMPT = """You are a helpful weather assistant. Based on the weather data provided, give a clear and friendly response to the user's question.

Weather Data:
{weather_data}

User Question: {query}

Provide a helpful, conversational response about the weather. Include relevant details from the data but present them in a natural way."""

RAG_RESPONSE_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Context from documents:
{context}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer, say so
3. Cite the source documents when possible
4. Be concise but thorough

Provide a helpful response:"""


class AgentNodes:
    """Container for LangGraph agent nodes."""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        weather_service: WeatherService,
        rag_retriever: RAGRetriever,
    ):
        """Initialize agent nodes.
        
        Args:
            llm: Language model for responses.
            weather_service: Weather API service.
            rag_retriever: RAG retriever for document queries.
        """
        self.llm = llm
        self.weather_service = weather_service
        self.rag_retriever = rag_retriever
    
    def router_node(self, state: dict) -> dict:
        """Route the query to the appropriate handler.
        
        Determines whether the query is about weather, documents, or is a greeting.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with query_type set.
        """
        query = state.get("query", "")
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        
        response = self.llm.invoke(messages)
        query_type = response.content.strip().lower()
        
        # Normalize the response
        if "weather" in query_type:
            query_type = "weather"
        elif "greeting" in query_type:
            query_type = "greeting"
        else:
            query_type = "document"
        
        return {
            **state,
            "query_type": query_type,
        }
    
    def weather_node(self, state: dict) -> dict:
        """Fetch weather data and generate response.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with weather context.
        """
        query = state.get("query", "")
        
        # Extract city from query using LLM
        city_extraction_prompt = """Extract the city name from this weather query. 
Return ONLY the city name, nothing else. If no specific city is mentioned, return "New York".

Query: {query}

City:"""
        
        messages = [
            HumanMessage(content=city_extraction_prompt.format(query=query)),
        ]
        
        city_response = self.llm.invoke(messages)
        city = city_response.content.strip()
        
        try:
            # Get current weather
            weather_data = self.weather_service.get_current_weather(city)
            weather_summary = weather_data.to_summary()
            
            # Check if forecast is requested
            forecast_keywords = ["forecast", "tomorrow", "next", "week", "days", "upcoming"]
            if any(keyword in query.lower() for keyword in forecast_keywords):
                forecast_data = self.weather_service.get_forecast(city)
                weather_summary += "\n\n" + forecast_data.to_summary()
            
            context = weather_summary
            error = None
            
        except WeatherServiceError as e:
            context = f"Error fetching weather data: {str(e)}"
            error = str(e)
        
        return {
            **state,
            "context": context,
            "error": error,
            "source": "weather_api",
        }
    
    def rag_node(self, state: dict) -> dict:
        """Retrieve relevant documents and build context.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with document context.
        """
        query = state.get("query", "")
        
        try:
            # Retrieve relevant documents
            result = self.rag_retriever.retrieve(query, top_k=5)
            
            if result.num_results == 0:
                context = "No relevant documents found for your query."
            else:
                context = result.context
            
            sources = result.get_sources()
            
            return {
                **state,
                "context": context,
                "sources": sources,
                "num_results": result.num_results,
                "source": "documents",
                "error": None,
            }
            
        except Exception as e:
            return {
                **state,
                "context": f"Error retrieving documents: {str(e)}",
                "error": str(e),
                "source": "documents",
            }
    
    def greeting_node(self, state: dict) -> dict:
        """Handle greetings and casual conversation.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with greeting response.
        """
        query = state.get("query", "")
        
        # Generate a friendly greeting response
        greeting_responses = {
            "hello": "Hello! 👋 I'm your Weather RAG Assistant. I can help you with:\n\n🌤️ **Weather queries** - Ask about current weather or forecasts for any location\n📄 **Document questions** - Learn about weather forecasting, climate data, and more\n\nWhat would you like to know?",
            "hi": "Hi there! 👋 I'm here to help you with weather information and answer questions about weather forecasting. What can I help you with today?",
            "hey": "Hey! 👋 Ready to help with weather data or answer your questions about weather forecasting. What's on your mind?",
            "good morning": "Good morning! ☀️ How can I assist you with weather information today?",
            "good afternoon": "Good afternoon! 🌤️ What weather information can I help you with?",
            "good evening": "Good evening! 🌙 How can I help you today?",
            "how are you": "I'm doing great, thank you for asking! 😊 I'm here to help you with weather queries and questions about weather forecasting. What would you like to know?",
        }
        
        # Check for common greetings
        query_lower = query.lower().strip()
        for greeting, response_text in greeting_responses.items():
            if greeting in query_lower:
                return {
                    **state,
                    "context": "greeting",
                    "response": response_text,
                    "source": "greeting",
                }
        
        # Default friendly response for other greetings
        default_response = "Hello! 👋 I'm your Weather RAG Assistant. I can help you check weather conditions anywhere in the world or answer questions about weather forecasting and climate data. What would you like to know?"
        
        return {
            **state,
            "context": "greeting",
            "response": default_response,
            "source": "greeting",
        }
    
    def response_node(self, state: dict) -> dict:
        """Generate the final response using the LLM.
        
        Args:
            state: Current graph state.
            
        Returns:
            Updated state with response.
        """
        query = state.get("query", "")
        context = state.get("context", "")
        query_type = state.get("query_type", "document")
        
        if state.get("error"):
            response = f"I apologize, but I encountered an error: {state['error']}"
        else:
            # Select appropriate prompt
            if query_type == "weather":
                prompt = WEATHER_RESPONSE_PROMPT.format(
                    weather_data=context,
                    query=query,
                )
            else:
                prompt = RAG_RESPONSE_PROMPT.format(
                    context=context,
                    query=query,
                )
            
            messages = [HumanMessage(content=prompt)]
            response_obj = self.llm.invoke(messages)
            response = response_obj.content
        
        return {
            **state,
            "response": response,
        }


def create_router_condition(state: dict) -> Literal["weather", "document", "greeting"]:
    """Routing condition for the graph.
    
    Args:
        state: Current graph state.
        
    Returns:
        Route name based on query type.
    """
    return state.get("query_type", "document")
