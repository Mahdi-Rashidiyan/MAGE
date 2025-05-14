import os
import json
from typing import Dict, Any, List, Optional, Union

# Template management for the MAGE framework
class PromptTemplate:
    """Class to manage prompt templates for the MAGE framework."""
    
    def __init__(self, template: str):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with placeholders for variables
        """
        self.template = template
        
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided variables.
        
        Args:
            **kwargs: Variables to fill in the template
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)


class PromptLibrary:
    """Library of prompt templates for the MAGE framework."""
    
    def __init__(self):
        """Initialize the prompt library with default templates."""
        self.templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """
        Load the default set of prompt templates.
        
        Returns:
            Dictionary of prompt templates
        """
        templates = {}
        
        # Specification Agent Templates
        templates["spec_refinement"] = PromptTemplate("""
You are a specialized hardware design agent focusing on specification refinement. Your task is to refine, structure,
and complete the following hardware specification. Add necessary details that might be missing but are standard for 
such components. Organize the specification in a clear structure with the following sections:

1. Overview: A brief description of the module
2. Interface Definition: Detailed list of all inputs, outputs, and their purposes
3. Functional Requirements: What the module should do
4. Design Constraints: Any constraints on timing, area, power, etc.
5. Parameters: Any configurable parameters of the module

Original Specification:
{specification}

Please provide a refined, detailed, and structured specification.
        """.strip())
        
        # Architecture Agent Templates
        templates["architecture_design"] = PromptTemplate("""
You are a specialized hardware architecture agent. Based on the following specification, design 
a high-level architecture for the RTL implementation. Include:

1. Module Hierarchy: The hierarchy of modules needed
2. Interface Definitions: Detailed interfaces between modules
3. Block Diagram: A textual representation of the block diagram
4. State Machines: Description of required state machines
5. Register/Memory: Key registers and memories needed
6. Clock Domains: Description of clock domains if needed
7. Datapath: Description of the datapath

Specification:
{refined_spec}

Please provide a detailed architecture design.
        """.strip())
        
        # RTL Coding Agent Templates
        templates["rtl_code_generation_verilog"] = PromptTemplate("""
You are a specialized RTL coding agent with expertise in Verilog. Based on the following architecture
description, write complete, synthesizable Verilog code. Follow these guidelines:

1. Write modular, reusable code
2. Include comprehensive comments
3. Follow best practices for synthesis
4. Define clear interfaces between modules
5. Include parameter definitions for configurable elements
6. Address edge cases and reset conditions
7. Consider clock domain crossing if applicable
8. Optimize for area and performance where applicable

Architecture Description:
{architecture}

Please provide the complete Verilog code implementation.
        """.strip())
        
        templates["rtl_code_generation_vhdl"] = PromptTemplate("""
You are a specialized RTL coding agent with expertise in VHDL. Based on the following architecture
description, write complete, synthesizable VHDL code. Follow these guidelines:

1. Write modular, reusable code
2. Include comprehensive comments
3. Follow best practices for synthesis
4. Define clear interfaces between modules
5. Include generic definitions for configurable elements
6. Address edge cases and reset conditions
7. Consider clock domain crossing if applicable
8. Optimize for area and performance where applicable

Architecture Description:
{architecture}

Please provide the complete VHDL code implementation.
        """.strip())
        
        # Verification Agent Templates
        templates["testbench_generation_verilog"] = PromptTemplate("""
You are a specialized verification agent. Create a comprehensive testbench in Verilog
for the following RTL code. The testbench should verify that the implementation meets the specification.
Include:

1. A self-checking testbench with automated pass/fail indication
2. Comprehensive test vectors covering normal operation, edge cases, and error conditions
3. Clear reporting of test results
4. Coverage analysis approach
5. Clock and reset generation
6. Assertions for critical properties
7. Proper simulation termination

Specification:
{spec}

RTL Code:
{rtl_code}

Please provide a complete Verilog testbench code.
        """.strip())
        
        templates["testbench_generation_vhdl"] = PromptTemplate("""
You are a specialized verification agent. Create a comprehensive testbench in VHDL
for the following RTL code. The testbench should verify that the implementation meets the specification.
Include:

1. A self-checking testbench with automated pass/fail indication
2. Comprehensive test vectors covering normal operation, edge cases, and error conditions
3. Clear reporting of test results
4. Coverage analysis approach
5. Clock and reset generation
6. Assertions for critical properties
7. Proper simulation termination

Specification:
{spec}

RTL Code:
{rtl_code}

Please provide a complete VHDL testbench code.
        """.strip())
        
        # Critic Agent Templates
        templates["rtl_code_review_verilog"] = PromptTemplate("""
You are a specialized RTL critic agent. Review the following Verilog code against the specification.
Identify issues in these categories:

1. Functional correctness: Does the code implement the specification correctly?
2. Synthesizability: Are there any non-synthesizable constructs?
3. Code style: Does the code follow good coding practices?
4. Performance: Are there any performance bottlenecks?
5. Power: Are there any issues that might increase power consumption?
6. Reusability: How reusable is the code?
7. Testability: Is the code easy to test?

For each issue found, provide:
- The exact line or block of code with the issue
- A detailed explanation of the issue
- A suggested fix

Specification:
{spec}

RTL Code:
{rtl_code}

Please provide:
1. A detailed list of issues found (categorized)
2. An improved version of the RTL code with all issues fixed
        """.strip())
        
        templates["rtl_code_review_vhdl"] = PromptTemplate("""
You are a specialized RTL critic agent. Review the following VHDL code against the specification.
Identify issues in these categories:

1. Functional correctness: Does the code implement the specification correctly?
2. Synthesizability: Are there any non-synthesizable constructs?
3. Code style: Does the code follow good coding practices?
4. Performance: Are there any performance bottlenecks?
5. Power: Are there any issues that might increase power consumption?
6. Reusability: How reusable is the code?
7. Testability: Is the code easy to test?

For each issue found, provide:
- The exact line or block of code with the issue
- A detailed explanation of the issue
- A suggested fix

Specification:
{spec}

RTL Code:
{rtl_code}

Please provide:
1. A detailed list of issues found (categorized)
2. An improved version of the RTL code with all issues fixed
        """.strip())
        
        # Advanced templates for system-level design
        templates["system_integration"] = PromptTemplate("""
You are a specialized system integration agent. For the following RTL modules, 
create a top-level integration module that connects them properly.

Modules to integrate:
{modules}

System requirements:
{requirements}

Please provide:
1. A detailed explanation of the integration approach
2. The complete RTL code for the top-level integration module
        """.strip())
        
        # Optimization agent
        templates["rtl_optimization"] = PromptTemplate("""
You are a specialized RTL optimization agent. Optimize the following RTL code 
for {optimization_target} (area, performance, power) while maintaining its functionality.

Original RTL Code:
{rtl_code}

Optimization target: {optimization_target}
Constraints: {constraints}

Please provide:
1. An analysis of optimization opportunities
2. The optimized RTL code
3. Expected improvements in {optimization_target}
        """.strip())
        
        return templates
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(template_name)
    
    def add_template(self, name: str, template: Union[str, PromptTemplate]) -> None:
        """
        Add a new template to the library.
        
        Args:
            name: Name of the template
            template: Template string or PromptTemplate object
        """
        if isinstance(template, str):
            template = PromptTemplate(template)
        self.templates[name] = template
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the templates to a JSON file.
        
        Args:
            filepath: Path to save the templates to
        """
        templates_dict = {name: template.template for name, template in self.templates.items()}
        with open(filepath, 'w') as f:
            json.dump(templates_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PromptLibrary':
        """
        Load templates from a JSON file.
        
        Args:
            filepath: Path to load the templates from
            
        Returns:
            New PromptLibrary instance with loaded templates
        """
        library = cls()
        with open(filepath, 'r') as f:
            templates_dict = json.load(f)
        
        for name, template in templates_dict.items():
            library.add_template(name, template)
        
        return library


class PromptManager:
    """Manager for handling prompts in the MAGE framework."""
    
    def __init__(self, library: Optional[PromptLibrary] = None):
        """
        Initialize a prompt manager.
        
        Args:
            library: PromptLibrary instance to use, creates a new one if None
        """
        self.library = library if library else PromptLibrary()
        
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a formatted prompt.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to fill in the template
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If template not found
        """
        template = self.library.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.format(**kwargs)
    
    def customize_prompt_for_model(self, prompt: str, model_name: str) -> str:
        """
        Customize a prompt for a specific model.
        
        Args:
            prompt: Original prompt
            model_name: Name of the model
            
        Returns:
            Customized prompt
        """
        # Adjust prompt based on model capabilities
        if "gpt" in model_name:
            # Add GPT-specific instructions
            prompt = f"As an assistant specialized in hardware design and RTL code generation:\n\n{prompt}"
        elif "llama" in model_name:
            # Add Llama-specific instructions
            prompt = f"[INST] {prompt} [/INST]"
        elif "phi" in model_name:
            # Add Phi-specific instructions
            prompt = f"<|system|>\nYou are a hardware design assistant specializing in RTL code generation.\n<|user|>\n{prompt}\n<|assistant|>"
        
        return prompt


# Example RTL snippets for few-shot learning
class RTLExamples:
    """Collection of RTL examples for few-shot learning."""
    
    @staticmethod
    def get_verilog_counter_example() -> str:
        """Get a simple Verilog counter example."""
        return """
module counter #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [WIDTH-1:0] count
);

    // Counter logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {WIDTH{1'b0}};
        end else if (enable) begin
            count <= count + 1'b1;
        end
    end

endmodule
        """.strip()
    
    @staticmethod
    def get_vhdl_counter_example() -> str:
        """Get a simple VHDL counter example."""
        return """
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity counter is
    generic (
        WIDTH : integer := 8
    );
    port (
        clk     : in  std_logic;
        rst_n   : in  std_logic;
        enable  : in  std_logic;
        count   : out std_logic_vector(WIDTH-1 downto 0)
    );
end entity counter;

architecture rtl of counter is
    signal count_reg : unsigned(WIDTH-1 downto 0);
begin
    -- Counter process
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            count_reg <= (others => '0');
        elsif rising_edge(clk) then
            if enable = '1' then
                count_reg <= count_reg + 1;
            end if;
        end if;
    end process;
    
    -- Output assignment
    count <= std_logic_vector(count_reg);
end architecture rtl;
        """.strip()
    
    @staticmethod
    def get_verilog_fifo_example() -> str:
        """Get a simple Verilog FIFO example."""
        return """
module fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire write_en,
    input wire read_en,
    input wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH-1:0] data_out,
    output wire empty,
    output wire full
);

    // Calculate required address width
    localparam ADDR_WIDTH = $clog2(DEPTH);
    
    // Internal registers
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_WIDTH-1:0] write_ptr;
    reg [ADDR_WIDTH-1:0] read_ptr;
    reg [ADDR_WIDTH:0] count;  // Extra bit to distinguish full from empty
    
    // FIFO status
    assign empty = (count == 0);
    assign full = (count == DEPTH);
    
    // Read logic
    assign data_out = mem[read_ptr];
    
    // Write pointer logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= {ADDR_WIDTH{1'b0}};
        end else if (write_en && !full) begin
            mem[write_ptr] <= data_in;
            write_ptr <= (write_ptr == DEPTH-1) ? {ADDR_WIDTH{1'b0}} : write_ptr + 1'b1;
        end
    end
    
    // Read pointer logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_ptr <= {ADDR_WIDTH{1'b0}};
        end else if (read_en && !empty) begin
            read_ptr <= (read_ptr == DEPTH-1) ? {ADDR_WIDTH{1'b0}} : read_ptr + 1'b1;
        end
    end
    
    // Count logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {(ADDR_WIDTH+1){1'b0}};
        end else begin
            case ({write_en && !full, read_en && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule
        """.strip()
    
    @staticmethod
    def get_vhdl_fifo_example() -> str:
        """Get a simple VHDL FIFO example."""
        return """
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity fifo is
    generic (
        DATA_WIDTH : integer := 8;
        DEPTH : integer := 16
    );
    port (
        clk         : in  std_logic;
        rst_n       : in  std_logic;
        write_en    : in  std_logic;
        read_en     : in  std_logic;
        data_in     : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        data_out    : out std_logic_vector(DATA_WIDTH-1 downto 0);
        empty       : out std_logic;
        full        : out std_logic
    );
end entity fifo;

architecture rtl of fifo is
    -- Calculate required address width
    constant ADDR_WIDTH : integer := integer(ceil(log2(real(DEPTH))));
    
    -- Memory and pointers
    type memory_type is array (0 to DEPTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mem : memory_type;
    signal write_ptr : unsigned(ADDR_WIDTH-1 downto 0);
    signal read_ptr : unsigned(ADDR_WIDTH-1 downto 0);
    signal count : unsigned(ADDR_WIDTH downto 0);
    
    -- Internal signals
    signal empty_i : std_logic;
    signal full_i : std_logic;
begin
    -- FIFO status
    empty_i <= '1' when count = 0 else '0';
    full_i <= '1' when count = DEPTH else '0';
    
    empty <= empty_i;
    full <= full_i;
    
    -- Read output
    data_out <= mem(to_integer(read_ptr)) when empty_i = '0' else (others => '0');
    
    -- Write process
    write_proc: process(clk, rst_n)
    begin
        if rst_n = '0' then
            write_ptr <= (others => '0');
        elsif rising_edge(clk) then
            if write_en = '1' and full_i = '0' then
                mem(to_integer(write_ptr)) <= data_in;
                if write_ptr = DEPTH-1 then
                    write_ptr <= (others => '0');
                else
                    write_ptr <= write_ptr + 1;
                end if;
            end if;
        end if;
    end process;
    
    -- Read process
    read_proc: process(clk, rst_n)
    begin
        if rst_n = '0' then
            read_ptr <= (others => '0');
        elsif rising_edge(clk) then
            if read_en = '1' and empty_i = '0' then
                if read_ptr = DEPTH-1 then
                    read_ptr <= (others => '0');
                else
                    read_ptr <= read_ptr + 1;
                end if;
            end if;
        end if;
    end process;
    
    -- Count process
    count_proc: process(clk, rst_n)
    begin
        if rst_n = '0' then
            count <= (others => '0');
        elsif rising_edge(clk) then
            case write_en & read_en & full_i & empty_i is
                when "1000" | "1010" =>  -- Write only
                    count <= count + 1;
                when "0100" | "0101" =>  -- Read only
                    count <= count - 1;
                when "1100" | "1101" | "1110" =>  -- Read and write
                    count <= count;
                when others =>  -- No operation or invalid states
                    count <= count;
            end case;
        end if;
    end process;
end architecture rtl;
        """.strip()
    
    @staticmethod
    def get_example_by_name(name: str, language: str = "verilog") -> str:
        """
        Get an example by name and language.
        
        Args:
            name: Name of the example (counter, fifo, etc.)
            language: Language of the example (verilog, vhdl)
            
        Returns:
            Example code
        """
        language = language.lower()
        name = name.lower()
        
        if language == "verilog":
            if name == "counter":
                return RTLExamples.get_verilog_counter_example()
            elif name == "fifo":
                return RTLExamples.get_verilog_fifo_example()
        elif language == "vhdl":
            if name == "counter":
                return RTLExamples.get_vhdl_counter_example()
            elif name == "fifo":
                return RTLExamples.get_vhdl_fifo_example()
        
        raise ValueError(f"No example found for name '{name}' and language '{language}'")


class PromptEnhancer:
    """Tools for enhancing prompts with few-shot examples and chain of thought."""
    
    @staticmethod
    def add_few_shot_examples(prompt: str, examples: List[Dict[str, str]]) -> str:
        """
        Add few-shot examples to a prompt.
        
        Args:
            prompt: Original prompt
            examples: List of example dictionaries with 'input' and 'output' keys
            
        Returns:
            Enhanced prompt with examples
        """
        examples_text = "\n\nHere are some examples to guide you:\n\n"
        
        for i, example in enumerate(examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n\n"
        
        enhanced_prompt = prompt + examples_text
        return enhanced_prompt
    
    @staticmethod
    def add_chain_of_thought(prompt: str) -> str:
        """
        Add chain of thought instructions to a prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt with CoT instructions
        """
        cot_instruction = "\n\nThink step by step before providing your final answer. First analyze the requirements, then plan your approach, and finally implement the solution."
        
        enhanced_prompt = prompt + cot_instruction
        return enhanced_prompt
    
    @staticmethod
    def add_rtl_example(prompt: str, example_name: str, language: str = "verilog") -> str:
        """
        Add an RTL example to a prompt.
        
        Args:
            prompt: Original prompt
            example_name: Name of the example to add
            language: Language of the example
            
        Returns:
            Enhanced prompt with example
        """
        try:
            example = RTLExamples.get_example_by_name(example_name, language)
            
            examples_text = f"\n\nHere's an example of a well-written {language.upper()} {example_name} module to guide your implementation style:\n\n```{language}\n{example}\n```\n\n"
            
            enhanced_prompt = prompt + examples_text
            return enhanced_prompt
        except ValueError:
            # If example not found, return original prompt
            return prompt


# Utility functions for prompts
def get_optimized_prompt_manager() -> PromptManager:
    """
    Create and return an optimized PromptManager for laptop use.
    
    Returns:
        Optimized PromptManager instance
    """
    # Create the PromptLibrary
    library = PromptLibrary()
    
    # Create the PromptManager
    manager = PromptManager(library)
    
    return manager