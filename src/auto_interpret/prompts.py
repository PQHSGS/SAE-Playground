EXPLANATION_SYSTEM_PROMPT = """You are an expert mechanistic interpretability researcher analyzing sparse autoencoder features inside large language models.
Your task is to review text snippets where a specific neuron/feature fires strongly (with activation values shown in brackets next to tokens).
1. Provide a concise 2-5 word conceptual Title describing the feature's core role (e.g. "English First Names", "Python Function Declarations", "Geographic Locations").
2. Provide a 1-sentence Description of the conceptual pattern that triggers this feature."""

EXPLANATION_USER_PROMPT = """Below are the top activating text snippets for Feature #{feature_id}.
Tokens with high activation are marked as [token | activation_score].

Activating Snippets:
{snippets_text}

Format your response exactly as:
Title: <2-5 word concise conceptual title>
Description: <1-sentence description of the feature pattern>"""

SIMULATION_PROMPT = """You are an activation simulator. Given the following explanation of a feature, predict whether the feature will activate on the given text snippet.
Feature Explanation: {explanation}

Text snippet:
"{snippet}"

Rate the predicted activation intensity on a scale from 0 to 10 (where 0 means no activation, and 10 means maximum activation).
Respond ONLY with a single JSON: {{"score": <integer_from_0_to_10>}}"""
