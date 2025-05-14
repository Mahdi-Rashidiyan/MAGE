import os
import json
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Agent(ABC):
    """Base agent class that all specialized agents will inherit from."""
    
    def __init__(self, name: str, model_path: str = None, device: str = None):
        """
        Initialize an agent.
        
        Args:
            name: Name of the agent
            model_path: Path to the model weights (if applicable)
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.name = name
        self.model_path = model_path
        self.memory = []  # Agent memory to store context
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initialized agent {name} on device {self.device}")
    
    def add_to_memory(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the agent's memory."""
        self.memory.append(entry)
        
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory = []
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process the input data and return a result."""
        pass


class LLMAgent(Agent):
    """Agent that uses an LLM for processing."""
    
    def __init__(self, name: str, model_name: str = "gpt2", device: str = None):
        """
        Initialize an LLM agent.
        
        Args:
            name: Name of the agent
            model_name: Name or path of the model to use
            device: Device to run the model on
        """
        super().__init__(name, model_path=model_name, device=device)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 1024, temperature: float = 0.7, 
                     top_p: float = 0.9, top_k: int = 50) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()
    
    def process(self, input_data: str) -> str:
        """Process the input using the LLM."""
        return self.generate_text(input_data)


class SpecificationAgent(LLMAgent):
    """Agent responsible for understanding and refining the initial specification."""
    
    def __init__(self, name: str = "SpecificationAgent", model_name: str = "gpt2", device: str = None):
        super().__init__(name, model_name, device)
        
    def refine_specification(self, spec: str) -> str:
        """
        Refine and structure the input specification.
        
        Args:
            spec: Initial specification from the user
            
        Returns:
            Refined and structured specification
        """
        prompt = f"""You are a hardware design specification expert. Your task is to refine, structure, 
        and complete the following hardware specification. Add necessary details that might be missing 
        but are standard for such components, and organize the specification in a clear structure.
        
        Original Specification:
        {spec}
        
        Please provide a refined, detailed, and structured specification:"""
        
        return self.process(prompt)


class ArchitectureAgent(LLMAgent):
    """Agent responsible for designing the high-level architecture of the RTL module."""
    
    def __init__(self, name: str = "ArchitectureAgent", model_name: str = "gpt2", device: str = None):
        super().__init__(name, model_name, device)
        
    def design_architecture(self, refined_spec: str) -> str:
        """
        Design the high-level architecture based on the refined specification.
        
        Args:
            refined_spec: Refined specification from SpecificationAgent
            
        Returns:
            High-level architecture design
        """
        prompt = f"""You are a hardware architecture expert. Based on the following specification, 
        design a high-level architecture for the RTL implementation. Include module hierarchy, interfaces, 
        main internal registers, state machines, and any other important architectural elements.
        
        Specification:
        {refined_spec}
        
        Please provide a detailed architecture design:"""
        
        return self.process(prompt)


class RTLCodingAgent(LLMAgent):
    """Agent responsible for writing the actual RTL code."""
    
    def __init__(self, name: str = "RTLCodingAgent", model_name: str = "gpt2", device: str = None):
        super().__init__(name, model_name, device)
        
    def generate_rtl_code(self, architecture: str, language: str = "verilog") -> str:
        """
        Generate RTL code based on the architecture.
        
        Args:
            architecture: Architecture design from ArchitectureAgent
            language: RTL language to use (verilog, VHDL, etc.)
            
        Returns:
            RTL code implementation
        """
        prompt = f"""You are an expert RTL designer specializing in {language}. Based on the following 
        architecture description, write complete, synthesizable {language} code. Include all necessary 
        modules, signals, and comments.
        
        Architecture Description:
        {architecture}
        
        Please provide the complete {language} code:"""
        
        return self.process(prompt)


class VerificationAgent(LLMAgent):
    """Agent responsible for generating testbench and verification code."""
    
    def __init__(self, name: str = "VerificationAgent", model_name: str = "gpt2", device: str = None):
        super().__init__(name, model_name, device)
        
    def generate_testbench(self, rtl_code: str, spec: str, language: str = "verilog") -> str:
        """
        Generate a testbench for the RTL code.
        
        Args:
            rtl_code: The RTL code to verify
            spec: Original or refined specification
            language: RTL language used
            
        Returns:
            Testbench code
        """
        prompt = f"""You are an expert in hardware verification. Create a comprehensive testbench in {language} 
        for the following RTL code. The testbench should verify that the implementation meets the specification.
        
        Specification:
        {spec}
        
        RTL Code:
        {rtl_code}
        
        Please provide a complete testbench code:"""
        
        return self.process(prompt)


class CriticAgent(LLMAgent):
    """Agent responsible for reviewing and improving the RTL code."""
    
    def __init__(self, name: str = "CriticAgent", model_name: str = "gpt2", device: str = None):
        super().__init__(name, model_name, device)
        
    def review_rtl_code(self, rtl_code: str, spec: str, language: str = "verilog") -> Tuple[str, List[str]]:
        """
        Review the RTL code for issues and suggest improvements.
        
        Args:
            rtl_code: The RTL code to review
            spec: Original or refined specification
            language: RTL language used
            
        Returns:
            Tuple of (improved RTL code, list of issues found)
        """
        prompt = f"""You are an expert RTL code reviewer. Review the following {language} code against the specification.
        Identify any issues, bugs, optimizations, or improvements needed. Provide an improved version of the code
        and list all issues found.
        
        Specification:
        {spec}
        
        RTL Code:
        {rtl_code}
        
        Please provide:
        1. A list of issues found
        2. An improved version of the RTL code"""
        
        response = self.process(prompt)
        
        # Parse the response to extract issues and improved code
        # This is a simple parsing logic and might need to be improved
        try:
            parts = response.split("An improved version of the RTL code", 1)
            issues_part = parts[0].split("A list of issues found", 1)[1].strip()
            issues = [issue.strip() for issue in issues_part.split('\n') if issue.strip()]
            
            improved_code = parts[1].strip() if len(parts) > 1 else rtl_code
            
            return improved_code, issues
        except Exception as e:
            logger.error(f"Failed to parse critic response: {e}")
            return rtl_code, ["Failed to parse critic response"]


class MAGEFramework:
    """
    Multi-Agent Engine (MAGE) for Automated RTL Code Generation.
    Orchestrates the interaction between different specialized agents.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
        Initialize the MAGE framework.
        
        Args:
            model_name: Base model name to use for all agents
            device: Device to run on
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing MAGE framework on device: {self.device}")
        
        # Initialize all agents
        self.specification_agent = SpecificationAgent(model_name=model_name, device=self.device)
        self.architecture_agent = ArchitectureAgent(model_name=model_name, device=self.device)
        self.rtl_coding_agent = RTLCodingAgent(model_name=model_name, device=self.device)
        self.verification_agent = VerificationAgent(model_name=model_name, device=self.device)
        self.critic_agent = CriticAgent(model_name=model_name, device=self.device)
        
        # Store the artifacts generated during the process
        self.artifacts = {
            "original_spec": None,
            "refined_spec": None,
            "architecture": None,
            "rtl_code": None,
            "improved_rtl_code": None,
            "testbench": None,
            "issues": None
        }
        
    def generate_rtl(self, specification: str, rtl_language: str = "verilog", 
                    review_iterations: int = 1) -> Dict[str, Any]:
        """
        Generate RTL code from a high-level specification.
        
        Args:
            specification: High-level specification of the hardware module
            rtl_language: Target RTL language (default: verilog)
            review_iterations: Number of review-improve cycles
            
        Returns:
            Dictionary containing all generated artifacts
        """
        logger.info("Starting RTL generation process")
        
        # Store original specification
        self.artifacts["original_spec"] = specification
        
        # Step 1: Refine the specification
        logger.info("Step 1: Refining specification")
        refined_spec = self.specification_agent.refine_specification(specification)
        self.artifacts["refined_spec"] = refined_spec
        
        # Step 2: Design the architecture
        logger.info("Step 2: Designing architecture")
        architecture = self.architecture_agent.design_architecture(refined_spec)
        self.artifacts["architecture"] = architecture
        
        # Step 3: Generate RTL code
        logger.info("Step 3: Generating RTL code")
        rtl_code = self.rtl_coding_agent.generate_rtl_code(architecture, rtl_language)
        self.artifacts["rtl_code"] = rtl_code
        
        # Step 4: Review and improve the RTL code (iterative)
        improved_rtl = rtl_code
        all_issues = []
        
        for i in range(review_iterations):
            logger.info(f"Step 4.{i+1}: Reviewing and improving RTL code (iteration {i+1})")
            improved_rtl, issues = self.critic_agent.review_rtl_code(
                improved_rtl if i > 0 else rtl_code,
                refined_spec,
                rtl_language
            )
            all_issues.extend(issues)
        
        self.artifacts["improved_rtl_code"] = improved_rtl
        self.artifacts["issues"] = all_issues
        
        # Step 5: Generate testbench
        logger.info("Step 5: Generating testbench")
        testbench = self.verification_agent.generate_testbench(
            improved_rtl,
            refined_spec,
            rtl_language
        )
        self.artifacts["testbench"] = testbench
        
        logger.info("RTL generation process completed")
        return self.artifacts
    
    def save_artifacts(self, output_dir: str) -> None:
        """
        Save all generated artifacts to files.
        
        Args:
            output_dir: Directory to save artifacts to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text artifacts
        for name, content in self.artifacts.items():
            if content and isinstance(content, str):
                with open(os.path.join(output_dir, f"{name}.txt"), 'w') as f:
                    f.write(content)
            elif content and isinstance(content, list):
                with open(os.path.join(output_dir, f"{name}.json"), 'w') as f:
                    json.dump(content, f, indent=2)
        
        # Save a summary JSON
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "artifact_keys": list(self.artifacts.keys())
            }
            json.dump(summary, f, indent=2)
        
        logger.info(f"Artifacts saved to {output_dir}")


# Utility functions
def optimize_for_laptop(mage_framework: MAGEFramework) -> None:
    """
    Apply optimizations to make the framework run efficiently on a laptop.
    
    Args:
        mage_framework: The MAGE framework instance to optimize
    """
    # Set all agents to use half precision on CUDA if available
    if torch.cuda.is_available():
        logger.info("Applying half precision optimizations for CUDA")
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("Running on CPU, applying CPU optimizations")
        # Enable CPU optimizations
        torch.set_num_threads(6)  # Limiting threads for Core i7

def get_optimized_mage_instance(model_name: str = "gpt2") -> MAGEFramework:
    """
    Create and return an optimized MAGE instance for laptop use.
    
    Args:
        model_name: The model to use
        
    Returns:
        Optimized MAGE framework instance
    """
    # Determine the most appropriate model for laptop use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create the MAGE instance
    mage = MAGEFramework(model_name=model_name, device=device)
    
    # Apply optimizations
    optimize_for_laptop(mage)
    
    return mage