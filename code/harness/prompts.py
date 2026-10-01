"""Prompts for the harness arms.

The task statement (``modified_content``) is passed verbatim in every arm; only
the I/O contract around it differs. The ask policy and the two worked examples
are copied verbatim from ``clarifycodebench.prompts.SYSTEM_PROMPT``; the only
edit is the output-format paragraph, because the ``[QUESTION]``/``[CODE]`` tag
contract disables a harness's tool loop and would turn the harness back into a
plain chat model.
"""
from __future__ import annotations

# The native I/O contract. Kept minimal on purpose: it says where the program
# goes and nothing about whether to ask.
NATIVE_CONTRACT = """\
Write your solution to `solution.py` in the current directory, as a complete
standalone program that reads from standard input and prints to standard output.
Call finish when the file is written."""

# Verbatim from SYSTEM_PROMPT, minus the output-format paragraph.
ASK_POLICY = """\
You are a professional software developer. Your primary goal is to write Python code based on user requirements.
Before you start coding, you must perform a critical first step: Carefully evaluate the requirements.
1. If requirements are clear: Write the code directly.
2. If requirements are unclear: Do not guess. You ask one single, critical clarifying question."""

# Replaces the tag paragraph. Same two options, expressed as harness actions.
ASK_CONTRACT = """\
To ask a question, reply with the question as a plain message and take no other
action; you will receive an answer and can then continue.
To deliver code, write it to `solution.py` in the current directory, as a
complete standalone program that reads from standard input and prints to
standard output, then call finish."""

# Verbatim from SYSTEM_PROMPT's [EXAMPLE1]/[EXAMPLE2], with the tags around the
# code and the question removed and the action named instead. The programs are
# byte-identical to the originals: they also carry the expected code shape
# (a main() that is called), which the pass@1 executor depends on.
EXAMPLES = """\
[EXAMPLE1]
You are given two positive integers A and B.
Output the square of (A + B).
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

The requirement is clear. Write this to `solution.py` and call finish:

def main():
    A, B = map(int, input().split())
    print((A + B) ** 2)
main()
[/EXAMPLE1]

[EXAMPLE2]
You are given two positive integers A and B.
Output the square of A + B.
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

The requirement is ambiguous. Reply with this message and take no other action:

By "the square of A + B," do you mean (A+B)^{2} or A^{2}+B?
[/EXAMPLE2]"""

# The tag arm keeps SYSTEM_PROMPT byte-for-byte. It is inlined rather than
# imported, because the evaluation host does not have the clarifycodebench
# package installed and an ImportError there would fail every tag-arm cell.
# _verify_tag_prompt() checks the copy against the original whenever the
# package IS importable, so the two cannot drift apart silently.
_SYSTEM_PROMPT = """\
You are a professional software developer. Your primary goal is to write Python code based on user requirements.
Before you start coding, you must perform a critical first step: Carefully evaluate the requirements.
1. If requirements are clear: Write the code directly.
2. If requirements are unclear: Do not guess. You ask one single, critical clarifying question.
Your response must be in one of the following two formats, with no extra text.

Output Format:

[CODE]
{Your code here}
[/CODE]

or

[QUESTION]
{Your single question here}
[/QUESTION]

[EXAMPLE1]
You are given two positive integers A and B.
Output the square of (A + B).
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

[CODE]
def main():
    A, B = map(int, input().split())
    print((A + B) ** 2)
main()
[/CODE]
[/EXAMPLE1]

[EXAMPLE2]
You are given two positive integers A and B.
Output the square of A + B.
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

[QUESTION]
By "the square of A + B," do you mean (A+B)^{2} or A^{2}+B?
[/QUESTION]
[/EXAMPLE2]
"""

_SYSTEM_PROMPT_SHA256 = "30fd05991f5363b4d0e7b6f527abf92652decf6196228176298bb6734b53ef97"


def tag_system_prompt() -> str:
    return _SYSTEM_PROMPT


def verify_tag_prompt() -> str:
    """Return 'ok', 'unavailable', or raise if the inlined copy has drifted."""
    import hashlib

    if hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest() != _SYSTEM_PROMPT_SHA256:
        raise AssertionError("inlined SYSTEM_PROMPT does not match its recorded digest")
    try:
        from clarifycodebench.prompts import SYSTEM_PROMPT
    except ImportError:
        return "unavailable"
    if SYSTEM_PROMPT != _SYSTEM_PROMPT:
        raise AssertionError(
            "inlined SYSTEM_PROMPT differs from clarifycodebench.prompts.SYSTEM_PROMPT"
        )
    return "ok"


def first_message(arm: str, modified_content: str) -> str:
    """Build the opening user message for one arm."""
    if arm == "native":
        return f"{modified_content}\n\n{NATIVE_CONTRACT}"
    if arm == "ask_permitted":
        return f"{ASK_POLICY}\n\n{ASK_CONTRACT}\n\n{EXAMPLES}\n\n{modified_content}"
    if arm == "tag":
        # Tag arm sends the statement alone; the policy lives in the system slot.
        return modified_content
    if arm == "protocol":
        # ClarifyCodeBench's SYSTEM_PROMPT is the benchmark's top-level protocol,
        # so it is handed to the harness verbatim -- exactly as vanilla and
        # CONTRA receive it -- and the harness runs as a method underneath it.
        #
        # It goes in the first user message rather than the system slot because
        # OpenHands owns that slot: replacing it would strip the tool
        # descriptions and the agent would not know what it can do.
        #
        # The protocol's "[QUESTION]/[CODE], no extra text" instruction and the
        # harness's tool loop genuinely conflict. That conflict is not smoothed
        # over here: how the agent resolves it is a property of the harness and
        # is what this arm measures. Both outcomes are accepted downstream --
        # tags are parsed if present, solution.py is read otherwise.
        return f"{tag_system_prompt()}\n{modified_content}"
    raise ValueError(f"unknown arm: {arm}")


ARMS = ("native", "ask_permitted", "tag", "protocol")

# Environment replies, verbatim from clarifycodebench.interact.
FOLLOWUP = "\nBased on our current conversation, please ask a clarifying question or generate the code."
NOT_RELEVANT = "This question is not relevant to the current task."
ALREADY_ASKED = "This question has already been answered."

# Sent once the question budget is spent.
NO_MORE_ANSWERS = (
    "No further clarification is available. Please write your solution to "
    "`solution.py` now and call finish."
)

# Sent when the agent emits a message that is not a question and not code, so
# that a stalled turn does not silently consume the question budget.
NUDGE = (
    "Please either ask one clarifying question, or write your solution to "
    "`solution.py` and call finish."
)
