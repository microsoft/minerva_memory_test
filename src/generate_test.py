import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from utils import TASK_CLASSES

# Import all task modules
from task.search import *
from task.recall_and_edit import *
from task.match_and_compare import *
from task.spot_the_differences import *
from task.compute_on_sets_and_lists import *
from task.stateful_processing import *
from task.composite import *


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_memory_tests(
    output_dir: str, 
    task_category: Optional[str] = None, 
    task_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate LLM memory tests.
    
    Args:
        output_dir: Directory to save the generated tests.
        task_category: Category of tasks to generate tests for (optional).
        task_name: Specific task to generate tests for (optional).
    
    Returns:
        List of dictionaries containing information about generated tests.
        
    Raises:
        ValueError: If an invalid task category or output format is specified.
    """

    os.makedirs(output_dir, exist_ok=True)
    generated_tests = []
    
    # Filter categories if specified
    if task_category:
        if task_category not in TASK_CLASSES:
            raise ValueError(f"Unknown task category: {task_category}")
        categories = [task_category]
    else:
        categories = TASK_CLASSES.keys()

    # Generate tests for each category
    for category in categories:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        logger.info(f"Generating tasks for category: {category}")

        for task_class_info in tqdm(TASK_CLASSES[category], desc=f"Tasks in {category}"):
            try:
                # Handle both simple class references and parameterized classes
                if isinstance(task_class_info, dict):
                    task_class = task_class_info["class"]
                    params = task_class_info["params"]
                    task_instance = task_class(**params)
                else:
                    task_class = task_class_info
                    task_instance = task_class()
            
                # Skip if we're filtering by task name and this doesn't match
                if task_name and task_instance.task_name != task_name:
                    continue
                
                logger.info(f"Generating task: {task_instance.task_name} ({task_class.__name__})")
                
                # Set the output path for this task
                file_extension = ".jsonl"
                task_output_path = os.path.join(category_dir, f"{task_instance.task_name}{file_extension}")
                task_instance.task_data_filepath = task_output_path

                # Generate the task data
                task_data = task_instance.compile_task_data()

                # Save the task data to a file

                with open(task_output_path, "w") as f:
                    for entry in task_data:
                        f.write(json.dumps(entry) + "\n")

                logger.info(f"Saved {len(task_data)} samples to {task_output_path}")

                generated_tests.append({
                    "category": category,
                    "task_name": task_instance.task_name,
                    "class_name": task_class.__name__,
                    "samples": len(task_data),
                    "path": task_output_path
                })
            except Exception as e:
                logger.error(f"Error generating task {task_class.__name__}: {str(e)}")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()

    # Output summary of generated tests
    if generated_tests:
        logger.info(f"Generated {len(generated_tests)} test sets:")
        for test in generated_tests:
            logger.info(f"  {test['category']} - {test['task_name']} ({test['class_name']}): {test['samples']} samples")
    else:
        logger.warning("No tests were generated. Check your filters or task configurations.")

    return generated_tests


def main():
    """Parse command line arguments and run the test generator."""
    parser = argparse.ArgumentParser(
        description="Generate Minerva LLM Memory tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save the generated tests"
    )
    parser.add_argument(
        "--task_category", 
        type=str, 
        help="Generate tests only for this category"
    )
    parser.add_argument(
        "--task_name", 
        type=str, 
        help="Generate tests only for this specific task"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with detailed error information"
    )
    parser.add_argument(
        "--list-tasks", 
        action="store_true", 
        help="List all available tasks and exit"
    )
    args = parser.parse_args()

    # List available tasks if requested
    if args.list_tasks:
        print("Available task categories and tasks:")
        for category, tasks in TASK_CLASSES.items():
            print(f"\n{category}:")
            for task_info in tasks:
                if isinstance(task_info, dict):
                    task_class = task_info["class"]
                    params = task_info["params"]
                    task_name = task_class(**params).task_name
                    print(f"  - {task_name} ({task_class.__name__} with {params})")
                else:
                    task_name = task_info().task_name
                    print(f"  - {task_name} ({task_info.__name__})")
        sys.exit(0)

    # Generate tests
    if not args.output_dir:
        logger.error("Output directory must be specified with --output_dir")
        sys.exit(1)

    try:
        generate_memory_tests(
            output_dir=args.output_dir,
            task_category=args.task_category,
            task_name=args.task_name,
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()