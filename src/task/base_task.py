"""
Base Task module for LLM_Memory.

This module defines the Task abstract base class that all memory tests inherit from,
providing standard interfaces and utilities for test generation.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from task.context_utils import ContextGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Task:
    """Base class for all LLM memory evaluation tasks.
    
    This abstract class defines the interface that all memory test tasks must implement.
    It provides common utilities for test generation, context management, and data handling.
    
    Attributes:
        task_name: Unique identifier for this task.
        task_category: Category this task belongs to.
        task_instruction: Human-readable instruction for the task.
        num_samples: Number of test samples to generate.
        variables: Dictionary of parameters to vary across test samples.
        task_data: List of generated test entries.
        metrics: Dictionary of evaluation metrics for this task.
        WORDS: List of common words for context generation.
        task_data_filepath: Path where task data should be saved.
    """
    
    def __init__(self) -> None:
        """Initialize a new Task."""
        self.task_name = ""
        self.task_category = ""
        self.task_instruction = ""

        self.num_samples = 10

        self.variables: Dict[str, List[Any]] = {}
        self.task_data: List[Dict[str, Any]] = []

        self.metrics: Dict[str, Any] = {}

        # Access words list from ContextGenerator
        self.WORDS = ContextGenerator.WORDS

        self.task_data_filepath: Optional[str] = None

    def __repr__(self) -> str:
        """Return string representation of the task."""
        return f"{self.task_category}_{self.task_name}"

    def create_entry_id(self) -> str:
        """Generate a unique ID for a test entry.
        
        Returns:
            A UUID string for the test entry.
        """
        return str(uuid4())

    def compile_task_data(self) -> List[Dict[str, Any]]:
        """Generate all test samples for this task.
        
        This is the main method to implement in subclasses. It should:
        1. Generate appropriate contexts
        2. Create test entries with prompts and expected answers
        3. Return the complete dataset
        
        Returns:
            List of test entries.
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement compile_task_data()")

    def get_reference(self) -> Any:
        """Get the expected answer for a test instance.
        
        Returns:
            The reference answer(s) for evaluation.
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement get_reference()")

    def create_context_data(self, context_type: str, length: int = 4096, 
                           num_samples: int = 10) -> List[str]:
        """Create context data using the ContextGenerator.
        
        Args:
            context_type: Type of context to generate (e.g., "unique_words", "random_numbers").
            length: Maximum length of the context in tokens.
            num_samples: Number of context samples to generate.
            
        Returns:
            List of generated context strings.
        """
        context_generator = ContextGenerator()
        context_data = context_generator.generate_context(
            context_type, length, num_samples
        )
        return context_data