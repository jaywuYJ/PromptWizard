from promptwizard.glue.promptopt import GluePromptOpt
from promptwizard.glue.common.dataset_processor import DatasetSpecificProcessing
import os
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

# Set environment variables for Gemini
os.environ["MODEL_TYPE"] = "Gemini"
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"  # Replace with your actual API key

# Configuration paths
prompt_config_path = "configs/promptopt_config.yaml"
setup_config_path = "configs/setup_config.yaml"
llm_config_path = "configs/llm_config.yaml"

def signal_handler(signum, frame):
    logger.info("\nCleaning up and exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Initialize GluePromptOpt
try:
    logger.info("Initializing GluePromptOpt...")
    gp = GluePromptOpt(
        prompt_config_path=prompt_config_path,
        setup_config_path=setup_config_path,
        llm_config_path=llm_config_path,
        dataset_jsonl=None,
        data_processor=None
    )

    logger.info("Starting prompt optimization...")
    logger.info(f"Will perform {gp.prompt_opt_param.mutate_refine_iterations} iterations")
    logger.info(f"Each iteration will generate {gp.prompt_opt_param.style_variation} variations")
    
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=False,
        run_without_train_examples=True,
        generate_synthetic_examples=False
    )
    
    if best_prompt and expert_profile:
        logger.info("\nOptimization completed successfully!")
        logger.info("-" * 80)
        logger.info("Expert Profile:")
        logger.info(expert_profile)
        logger.info("\nBest Prompt:")
        logger.info(best_prompt)
        logger.info("-" * 80)
    else:
        logger.warning("No prompt or expert profile generated")
    
except Exception as e:
    logger.error(f"\nTest failed with error: {str(e)}")
    raise e
finally:
    logger.info("Test completed") 