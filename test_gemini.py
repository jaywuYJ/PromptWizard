from promptwizard.glue.promptopt import GluePromptOpt
from promptwizard.glue.common.dataset_processor import DatasetSpecificProcessing
import os
import logging
import signal
import sys
import google.generativeai as genai

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prompt_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy logs but keep errors
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

# Set environment variables for Gemini
os.environ["MODEL_TYPE"] = "Gemini"

# Check for API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not set. Please set it before running:\n"
        "export GOOGLE_API_KEY=your_api_key_here"
    )

logger.info("Testing Gemini API connection...")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, are you working?")
    if response and hasattr(response, 'text'):
        logger.info("Successfully connected to Gemini API")
    else:
        raise ValueError("Received invalid response from Gemini")
except Exception as e:
    logger.error(f"Failed to connect to Gemini API: {str(e)}")
    raise

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
    logger.info("Configuration:")
    logger.info(f"- Base instruction: {gp.prompt_opt_param.base_instruction}")
    logger.info(f"- Task description: {gp.prompt_opt_param.task_description}")
    
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=False,
        run_without_train_examples=True,
        generate_synthetic_examples=False
    )
    
    if best_prompt or expert_profile:
        logger.info("\nOptimization completed!")
        logger.info("-" * 80)
        if expert_profile:
            logger.info("Expert Profile:")
            logger.info(expert_profile)
        if best_prompt:
            logger.info("\nBest Prompt:")
            logger.info(best_prompt)
        logger.info("-" * 80)
    else:
        logger.warning("No prompt or expert profile generated")
    
except Exception as e:
    logger.error(f"\nTest failed with error: {str(e)}")
    logger.error("Stack trace:", exc_info=True)
    raise e
finally:
    logger.info("Test completed") 