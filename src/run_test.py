import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional
import yaml
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

from inference import Azure_LLM_API
from evaluate import evaluate_generation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_task_data(task_data_path: str) -> List[Dict[str, Any]]:
    """Load task data from a JSONL file.
    
    Args:
        task_data_path: Path to the task data file.
        
    Returns:
        List of task data entries.
    """
    data = []

    if not os.path.exists(task_data_path):
        logger.error(f"Task data file not found: {task_data_path}")
        return data
    
    try:
        with open(task_data_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {task_data_path}")
    except Exception as e:
        logger.error(f"Error loading task data from {task_data_path}: {e}")
    
    logger.info(f"Loaded {len(data)} examples from {task_data_path}")
    return data
    

def run_test(
    task_data: List[Dict[str, Any]], 
    llm_api: Any, 
    metrics: Optional[List[str]] = None,
    result_file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a memory test using the provided task data and LLM API.
    
    Args:
        task_data: List of task data entries.
        llm_api: Instance of the LLM API to use for inference.
        metrics: List of metrics to evaluate the results.
        result_file_path: Path to save the results.
        max_retries: Maximum number of retries for failed API calls.
        
    Returns:
        List of results from the test.
    """
    results = []
    
    # Create the directory if it doesn't exist
    if result_file_path:
        os.makedirs(os.path.dirname(os.path.abspath(result_file_path)), exist_ok=True)
        result_file = open(result_file_path, 'w')
    else:
        result_file = None

    try:
        for entry in tqdm(task_data, desc="Processing entries"):
            entry_id = entry.get('id', 'unknown')
            try:
                prompt = entry.get("prompt", "")
                if not prompt:
                    logger.warning(f"No prompt found for entry {entry_id}. Skipping.")
                    break

                generation = llm_api.generate(prompt)

                result = entry.copy()
                result["generation"] = generation
                result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                if metrics:
                    result["scores"] = {}
                    for metric in metrics:
                        score = evaluate_generation(
                            generation, 
                            entry.get("reference", ""), 
                            metrics=[metric]
                        )
                        result["scores"][metric] = score

                results.append(result)
                if result_file:
                    result_file.write(json.dumps(result) + "\n")
                    result_file.flush()
                    
            except Exception as e:
                logger.error(f"Error processing entry {entry_id}: {e}")

    
    finally:
        if result_file:
            result_file.close()
    
    return results


def run_memory_tests(
    task_dir: str, 
    result_dir: str, 
    llm_api: Any, 
    model_name: str, 
    task_category: Optional[str] = None, 
    task_name: Optional[str] = None
) -> Dict[str, Any]:
    """Run LLM memory tests and save results.

    Args:
        task_dir: Directory containing the test data.
        result_dir: Directory to save the test results.
        llm_api: LLM API instance to use for inference.
        model_name: Name of the model being tested.
        task_category: Optional category of tasks to run.
        task_name: Optional specific task to run.
        
    Returns:
        Dictionary with summary of test results.
    """
    model_result_dir = os.path.join(result_dir, model_name)
    os.makedirs(model_result_dir, exist_ok=True)
    
    # Track overall statistics
    summary = {
        "model": model_name,
        "tasks_run": 0,
        "examples_total": 0,
        "examples_completed": 0,
        "categories": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Find all available categories
    categories_to_run = []
    for category in TASK_CLASSES.keys():
        if not task_category or category == task_category:
            categories_to_run.append(category)
    
    if task_category and not categories_to_run:
        logger.error(f"Task category '{task_category}' not found")
        return summary
        
    # Run tests for each category
    for category in categories_to_run:
        category_dir = os.path.join(task_dir, category)
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
            
        category_result_dir = os.path.join(model_result_dir, category)
        os.makedirs(category_result_dir, exist_ok=True)
        
        category_summary = {"tasks_run": 0, "examples_total": 0, "examples_completed": 0}
        summary["categories"][category] = category_summary

        # Get all task instances for this category
        task_instances = []
        for task_class_info in TASK_CLASSES[category]:
            try:
                if isinstance(task_class_info, dict):
                    task_class = task_class_info["class"]
                    params = task_class_info["params"]
                    task_instance = task_class(**params)
                else:
                    task_class = task_class_info
                    task_instance = task_class()
                    
                # Filter by task name if specified
                if task_name and task_instance.task_name != task_name:
                    continue
                    
                task_instances.append(task_instance)
            except Exception as e:
                logger.error(f"Error instantiating task {task_class.__name__}: {e}")
        
        # Run each task
        for task_instance in task_instances:
            logger.info(f"Running task: {task_instance.task_name}")
            
            task_data_path = os.path.join(category_dir, f"{task_instance.task_name}.jsonl")
            if not os.path.exists(task_data_path):
                logger.warning(f"Data file not found: {task_data_path}. Skipping.")
                continue
                
            # Load task data
            task_data = load_task_data(task_data_path)
            if not task_data:
                logger.warning(f"No data loaded for task: {task_instance.task_name}. Skipping.")
                continue

            # Run the test
            result_file_path = os.path.join(category_result_dir, f"{task_instance.task_name}_results.jsonl")
            start_time = time.time()
            
            results = run_test(
                task_data, 
                llm_api, 
                metrics=task_instance.metrics, 
                result_file_path=result_file_path
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Completed {len(results)}/{len(task_data)} examples for '{task_instance.task_name}' in {elapsed:.2f}s")
            
            # Update statistics
            summary["tasks_run"] += 1
            summary["examples_total"] += len(task_data)
            summary["examples_completed"] += len(results)
            
            category_summary["tasks_run"] += 1
            category_summary["examples_total"] += len(task_data)
            category_summary["examples_completed"] += len(results)
    
    # Save summary
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["duration_seconds"] = time.time() - time.mktime(time.strptime(summary["start_time"], "%Y-%m-%d %H:%M:%S"))
    
    summary_path = os.path.join(model_result_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Testing complete: {summary['examples_completed']}/{summary['examples_total']} examples across {summary['tasks_run']} tasks")
    logger.info(f"Summary saved to {summary_path}")
    
    return summary


def main():
    """Parse arguments and run the memory test suite."""
    parser = argparse.ArgumentParser(
        description="Run LLM Memory tests and evaluate model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task_dir", type=str, required=True, 
                        help="Directory containing test data files")
    parser.add_argument("--result_dir", type=str, required=True, 
                        help="Directory to save test results")
    parser.add_argument("--model_name", type=str, default="gpt-4o", 
                        help="Name of the model being tested")
    parser.add_argument("--llm_api_config", type=str, default="src/azure_api_config.yaml", 
                        help="Path to API configuration YAML file")
    parser.add_argument("--task_category", type=str, 
                        help="Run only tasks in this category")
    parser.add_argument("--task_name", type=str, 
                        help="Run only this specific task")
    parser.add_argument("--list-tasks", action="store_true", 
                        help="List available task categories and names, then exit")
    args = parser.parse_args()

    # List all available tasks if requested
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

    # Validate input directories
    if not os.path.exists(args.task_dir):
        logger.error(f"Task directory not found: {args.task_dir}")
        sys.exit(1)
    
    # Load API configuration
    if not os.path.exists(args.llm_api_config):
        logger.error(f"API configuration file not found: {args.llm_api_config}")
        sys.exit(1)
        
    try:
        with open(args.llm_api_config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load API configuration: {e}")
        sys.exit(1)

    # Use specified model name or fallback to config
    model_name = args.model_name or config.get("model_name", "gpt-4o")
    
    # Initialize the API
    try:
        llm_api = Azure_LLM_API(
            model_name=model_name, 
            endpoint=config.get("endpoint"),
            api_version=config.get("api_version"),
            client_id=config.get("client_id"),
            scope=config.get("scope")
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM API: {e}")
        sys.exit(1)

    # Run the tests
    try:
        run_memory_tests(
            task_dir=args.task_dir,
            result_dir=args.result_dir,
            llm_api=llm_api,
            model_name=model_name,
            task_category=args.task_category,
            task_name=args.task_name
        )
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()