
import sys
import os
from pathlib import Path

# Add src directory to Python path to import from main package
project_root = Path(__file__).parent.parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

# Clear module cache to ensure fresh imports from main package
for module_name in list(sys.modules.keys()):
    if module_name.startswith('email_assistant'):
        del sys.modules[module_name]

import pytest
from email_assistant.eval.email_dataset import email_inputs, expected_tool_calls
from email_assistant.utils import format_messages_string
from email_assistant.email_assistant import email_assistant
from email_assistant.utils import extract_tool_calls
from langsmith import testing as t
from dotenv import load_dotenv

load_dotenv(".env")

@pytest.mark.langsmith
@pytest.mark.parametrize(
    "email_input, expected_calls",
    [   # Pick some examples with e-mail reply expected
        (email_inputs[0],expected_tool_calls[0]),
        (email_inputs[3],expected_tool_calls[3]),
    ],
)
def test_email_dataset_tool_calls(email_input, expected_calls):
    """Test if email processing contains expected tool calls."""    
    # Run the email assistant
    result = email_assistant.invoke({"email_input": email_input})
            
    # Extract tool calls from messages list
    extracted_tool_calls = extract_tool_calls(result['messages'])
            
    # Check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]
    
    t.log_outputs({
                "missing_calls": missing_calls,
                "extracted_tool_calls": extracted_tool_calls,
                "response": format_messages_string(result['messages'])
            })

    # Test passes if no expected calls are missing
    assert len(missing_calls) == 0
