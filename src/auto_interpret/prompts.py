EXPLANATION_SYSTEM_PROMPT = """You are an expert mechanist interpretability researcher analyzing features inside large language models.
Your task is to review text snippets where a specific neuron/feature fires strongly (with activation values shown in brackets next to tokens).
Identify the core semantic, syntactic, or conceptual pattern that triggers this feature.
Provide a concise 1-2 sentence explanation of what concept, entity type, language, or context activates this feature."""

EXPLANATION_USER_PROMPT = """Below are the top activating text snippets for Feature #{feature_id}.
Tokens with high activation are marked as [token | activation_score].

Activating Snippets:
{snippets_text}

Provide a concise, precise summary of what this feature represents:
Explanation:"""

SIMULATION_PROMPT = """You are an activation simulator. Given the following explanation of a feature, predict whether the feature will activate on the given text snippet.
Feature Explanation: {explanation}

Text snippet:
"{snippet}"

Rate the predicted activation intensity on a scale from 0 to 10 (where 0 means no activation, and 10 means maximum activation).
Respond ONLY with a single JSON: {{"score": <integer_from_0_to_10>}}"""
