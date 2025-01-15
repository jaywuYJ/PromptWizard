from typing import Dict, List
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, wait_exponential, retry_if_exception_type
from ..base_classes import LLMConfig
from ..constants.str_literals import InstallLibs, OAILiterals, \
    OAILiterals, LLMLiterals, LLMOutputTypes
from .llm_helper import get_token_counter
from ..exceptions import GlueLLMException
from ..utils.runtime_tasks import install_lib_if_missing
from ..utils.logging import get_glue_logger
from ..utils.runtime_tasks import str_to_class
import os
import google.generativeai as genai
import atexit
import signal
import time
logger = get_glue_logger(__name__)

def call_api(messages):

    from openai import OpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential
    from openai import AzureOpenAI

    llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
    
    if llm_handle == "AzureOpenAI":
        if os.environ['USE_OPENAI_API_KEY'] == "True":
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL_NAME"],
                messages=messages,
                temperature=0.0,
            )
        else:
            token_provider = get_bearer_token_provider(
                AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                api_version=os.environ["OPENAI_API_VERSION"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_ad_token_provider=token_provider
            )
            response = client.chat.completions.create(
                model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                messages=messages,
                temperature=0.0,
            )
        prediction = response.choices[0].message.content
        
    elif llm_handle == "Gemini":
        # Configure Gemini
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        
        # Convert OpenAI message format to Gemini format
        chat = model.start_chat()
        for msg in messages:
            if msg["role"] == "system":
                # Add system message as user message since Gemini doesn't have system messages
                chat.send_message(msg["content"])
            elif msg["role"] == "user":
                chat.send_message(msg["content"])
            elif msg["role"] == "assistant":
                # Simulate assistant messages in chat history
                chat.send_message(msg["content"])
                
        # Get response for the last user message
        response = chat.send_message(messages[-1]["content"])
        prediction = response.text
        
    else:
        raise ValueError(f"Unsupported model type: {llm_handle}")
        
    return prediction


class LLMMgr:
    """Manager class for LLM interactions"""
    
    _instance = None
    _initialized = False
    _config = None
    _model = None
    
    @classmethod
    def initialize(cls, config: Dict):
        """Initialize LLM configurations"""
        cls._config = config
        if os.environ.get("MODEL_TYPE") == "Gemini":
            try:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                    
                genai.configure(api_key=api_key)
                cls._model = genai.GenerativeModel('gemini-pro')
                logger.info("Successfully initialized Gemini configuration")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                raise
                
        cls._initialized = True

    @classmethod
    def _cleanup(cls):
        """Clean up resources"""
        if cls._model:
            logger.info("Cleaning up Gemini resources...")
            cls._model = None

    @classmethod
    def _signal_handler(cls, signum, frame):
        """Handle interruption signals"""
        cls._cleanup()
        signal.default_int_handler(signum, frame)

    @staticmethod
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(3))  # Exponential backoff: 2s, 4s, 8s...)
    
    def chat_completion(messages: Dict):
        """Handle chat completion requests with retry logic for rate limits."""
        llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
        logger.debug(f"Using LLM: {llm_handle}")
        
        try:
            if llm_handle == "Gemini":
                # Add delay between requests
                time.sleep(1)  # 1 second delay between requests
                
                try:
                    # Configure Gemini
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY not set")
                        
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # Convert messages to Gemini format
                    chat = model.start_chat()
                    system_message = ""
                    user_message = ""
                    
                    # Process messages in order
                    for msg in messages:
                        logger.debug(f"Processing message: {msg['role']}")
                        if msg["role"] == "system":
                            system_message += msg["content"] + "\n"
                        elif msg["role"] == "user":
                            user_message = msg["content"]
                    
                    # Combine system and user messages
                    final_prompt = f"{system_message}\n{user_message}" if system_message else user_message
                    logger.debug(f"Sending prompt to Gemini: {final_prompt[:200]}...")
                    
                    try:
                        response = model.generate_content(final_prompt)
                        if response and hasattr(response, 'text'):
                            logger.debug(f"Received response from Gemini: {response.text[:200]}...")
                            return response.text
                        else:
                            logger.warning("Empty or invalid response from Gemini")
                            return ""
                    except Exception as api_error:
                        logger.error(f"Gemini API error: {str(api_error)}")
                        raise
                        
                except Exception as e:
                    logger.error(f"Error in Gemini chat: {str(e)}")
                    raise
                    
            elif llm_handle == "AzureOpenAI":
                return call_api(messages)
            else:
                raise ValueError(f"Unsupported model type: {llm_handle}")
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            logger.error(f"Messages that caused error: {messages}")
            raise

    @staticmethod
    def get_all_model_ids_of_type(llm_config: LLMConfig, llm_output_type: str):
        res = []
        if llm_config.azure_open_ai:
            for azure_model in llm_config.azure_open_ai.azure_oai_models:
                if azure_model.model_type == llm_output_type:
                    res.append(azure_model.unique_model_id)
        if llm_config.custom_models:
            if llm_config.custom_models.model_type == llm_output_type:
                res.append(llm_config.custom_models.unique_model_id)
        return res

    @staticmethod
    def get_llm_pool(llm_config: LLMConfig) -> Dict[str, LLM]:
        """
        Create a dictionary of LLMs. key would be unique id of LLM, value is object using which
        methods associated with that LLM service can be called.

        :param llm_config: Object having all settings & preferences for all LLMs to be used in out system
        :return: Dict key=unique_model_id of LLM, value=Object of class llama_index.core.llms.LLM
        which can be used as handle to that LLM
        """
        llm_pool = {}
        az_llm_config = llm_config.azure_open_ai

        if az_llm_config:
            install_lib_if_missing(InstallLibs.LLAMA_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_EMB_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_MM_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.TIKTOKEN)

            import tiktoken
            # from llama_index.llms.azure_openai import AzureOpenAI
            from openai import AzureOpenAI
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

            az_token_provider = None
            # if az_llm_config.use_azure_ad:
            from azure.identity import get_bearer_token_provider, AzureCliCredential
            az_token_provider = get_bearer_token_provider(AzureCliCredential(),
                                                        "https://cognitiveservices.azure.com/.default")

            for azure_oai_model in az_llm_config.azure_oai_models:
                callback_mgr = None
                if azure_oai_model.track_tokens:
                    
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=tiktoken.encoding_for_model(azure_oai_model.model_name_in_azure).encode
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                    # ()

                if azure_oai_model.model_type in [LLMOutputTypes.CHAT, LLMOutputTypes.COMPLETION]:
                    # ()
                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAI(
                            # use_azure_ad=az_llm_config.use_azure_ad,
                                    azure_ad_token_provider=az_token_provider,
                                    # model=azure_oai_model.model_name_in_azure,
                                    # deployment_name=azure_oai_model.deployment_name_in_azure,
                                    api_key=az_llm_config.api_key,
                                    azure_endpoint=az_llm_config.azure_endpoint,
                                    api_version=az_llm_config.api_version,
                                    # callback_manager=callback_mgr
                                    )
                    # ()
                elif azure_oai_model.model_type == LLMOutputTypes.EMBEDDINGS:
                    llm_pool[azure_oai_model.unique_model_id] =\
                        AzureOpenAIEmbedding(use_azure_ad=az_llm_config.use_azure_ad,
                                             azure_ad_token_provider=az_token_provider,
                                             model=azure_oai_model.model_name_in_azure,
                                             deployment_name=azure_oai_model.deployment_name_in_azure,
                                             api_key=az_llm_config.api_key,
                                             azure_endpoint=az_llm_config.azure_endpoint,
                                             api_version=az_llm_config.api_version,
                                             callback_manager=callback_mgr
                                             )
                elif azure_oai_model.model_type == LLMOutputTypes.MULTI_MODAL:

                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAIMultiModal(use_azure_ad=az_llm_config.use_azure_ad,
                                              azure_ad_token_provider=az_token_provider,
                                              model=azure_oai_model.model_name_in_azure,
                                              deployment_name=azure_oai_model.deployment_name_in_azure,
                                              api_key=az_llm_config.api_key,
                                              azure_endpoint=az_llm_config.azure_endpoint,
                                              api_version=az_llm_config.api_version,
                                              max_new_tokens=4096
                                              )

        if llm_config.custom_models:
            for custom_model in llm_config.custom_models:
                # try:
                custom_llm_class = str_to_class(custom_model.class_name, None, custom_model.path_to_py_file)

                callback_mgr = None
                if custom_model.track_tokens:
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=custom_llm_class.get_tokenizer()
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                llm_pool[custom_model.unique_model_id] = custom_llm_class(callback_manager=callback_mgr)
                # except Exception as e:
                    # raise GlueLLMException(f"Custom model {custom_model.unique_model_id} not loaded.", e)
        return llm_pool

    @staticmethod
    def get_tokens_used(llm_handle: LLM) -> Dict[str, int]:
        """
        For a given LLM, output the number of tokens used.

        :param llm_handle: Handle to a single LLM
        :return: Dict of token-type and count of tokens used
        """
        token_counter = get_token_counter(llm_handle)
        if token_counter:
            return {
                LLMLiterals.EMBEDDING_TOKEN_COUNT: token_counter.total_embedding_token_count,
                LLMLiterals.PROMPT_LLM_TOKEN_COUNT: token_counter.prompt_llm_token_count,
                LLMLiterals.COMPLETION_LLM_TOKEN_COUNT: token_counter.completion_llm_token_count,
                LLMLiterals.TOTAL_LLM_TOKEN_COUNT: token_counter.total_llm_token_count
                }
        return None

    @staticmethod
    def get_embedding(text: str, model_id: str) -> List[float]:
        """
        Get embeddings for the given text using the specified model.
        """
        llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
        try:
            if llm_handle == "AzureOpenAI":
                # Existing Azure OpenAI embedding code
                return call_embedding_api(text, model_id)
            elif llm_handle == "Gemini":
                # Configure Gemini
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                
                # Get embeddings using Gemini's embedding model
                embedding_model = genai.GenerativeModel('embedding-001')
                result = embedding_model.embed_content(
                    text,
                    task_type="retrieval_document"  # or "retrieval_query" for query embeddings
                )
                
                return result.embedding
                
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            logger.error(f"Text that caused error: {text[:100]}...")
            return []
