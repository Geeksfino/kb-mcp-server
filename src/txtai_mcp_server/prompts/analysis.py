"""
Analysis-related prompts for the txtai MCP server.
"""
from typing import List, Optional, Dict

from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent


def register_analysis_prompts(mcp: FastMCP) -> None:
    """Register analysis-related prompts with the MCP server."""
    
    @mcp.prompt()
    def analyze_text(text: str, task: str) -> List[PromptMessage]:
        """
        Create a prompt for text analysis tasks.
        
        Args:
            text: Text to analyze
            task: Analysis task (e.g., 'sentiment', 'entities', 'summary')
        """
        task_descriptions = {
            "sentiment": "Analyze the sentiment and emotional tone of this text.",
            "entities": "Identify and explain key entities (people, organizations, locations, etc.) in this text.",
            "summary": "Provide a concise summary of the main points in this text."
        }
        
        task_desc = task_descriptions.get(task, f"Perform {task} analysis on this text.")
        
        messages = [
            PromptMessage(
                role="system",
                content=TextContent(
                    type="text",
                    text=f"You are an expert at {task} analysis. {task_desc}"
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please analyze this text:\n\n{text}"
                )
            )
        ]
        
        return messages
    
    @mcp.prompt()
    def analyze_pipeline_output(
        input_text: str,
        output_text: str,
        pipeline_name: str
    ) -> List[PromptMessage]:
        """
        Create a prompt to analyze pipeline output.
        
        Args:
            input_text: Original input text
            output_text: Pipeline output text
            pipeline_name: Name of the pipeline used
        """
        messages = [
            PromptMessage(
                role="system",
                content=TextContent(
                    type="text",
                    text=f"You are an expert at analyzing {pipeline_name} pipeline results."
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Here is the original input:"
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=input_text
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"And here is the {pipeline_name} pipeline output:"
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=output_text
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Please analyze the pipeline's output and explain how it processed the input. Consider aspects like quality, accuracy, and any notable transformations."
                )
            )
        ]
        
        return messages
    
    @mcp.prompt()
    def analyze_model_performance(
        model_name: str,
        task_type: str,
        examples: List[Dict]  
    ) -> List[PromptMessage]:
        """
        Create a prompt to analyze model performance on specific tasks.
        
        Args:
            model_name: Name of the model
            task_type: Type of task (e.g., 'translation', 'summary')
            examples: List of input/output examples
        """
        # Format examples
        formatted_examples = "\n\n".join(
            f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
            for ex in examples
        )
        
        messages = [
            PromptMessage(
                role="system",
                content=TextContent(
                    type="text",
                    text=f"You are an expert at evaluating AI model performance for {task_type} tasks."
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please analyze the performance of the {model_name} model on these {task_type} examples:\n\n{formatted_examples}\n\nConsider aspects like output quality, consistency, and appropriateness for the task."
                )
            )
        ]
        
        return messages
