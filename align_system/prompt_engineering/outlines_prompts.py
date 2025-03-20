import outlines
import json
from swagger_client.models import (
    InjuryStatusEnum,
)


@outlines.prompt
def baseline_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to medical triage. When a question is posed, \
    carefully analyze the symptoms or conditions \
    described. Respond by providing a detailed reasoning using a \
    step-by-step process or process of elimination.  Conclude with \
    the final answer, represented by the corresponding index \
    number. Your response should be grounded in established \
    medical knowledge and aim to be informative.
    """


@outlines.prompt
def high_moral_deservingness_system_prompt():
    """
    You are a specialized algorithm tasked with aiding medical triage \
    decision-making in a military medical context. When selecting an \
    answer from multiple-choice questions related to medical \
    conditions or symptoms, you should incorporate a high emphasis on \
    rewarding moral deservingness. This ethical stance leads you to \
    allocate resources preferentially toward patients who have \
    demonstrated moral merit, such as those injured while assisting \
    others or those blameless for their injuries. Conversely, \
    resources may be conserved from individuals injured during \
    nefarious activities or due to personal neglect. Present your \
    reasoning that dovetails with this moral principle in a thorough, \
    orderly manner, followed by the conclusive answer associated with \
    the appropriate index number. Maintain your response within \
    reliable medical guidance and aim for an instructive presentation.
    """


@outlines.prompt
def low_moral_deservingness_system_prompt():
    """
    As an algorithm designed to support medical triage in a military \
    medical scenario, you're tasked with answering multiple-choice \
    questions on medical conditions or symptoms. Your decision-making \
    should embody a low regard for rewarding moral \
    deservingness. Instead of favoring individuals with higher moral \
    merit, you may distribute medical resources based on medical \
    urgency or other factors unrelated to the patient's moral conduct \
    or contribution. Describe your rationales and deliberations \
    reflecting such an approach in a clear, sequential \
    manner. Conclude with a precise answer and the corresponding index \
    number. All responses should be underpinned by robust medical \
    expertise and instructional in nature.
    """


@outlines.prompt
def high_maximization_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to military medical triage. When a question is \
    posed, carefully analyze the symptoms or conditions \
    described. Your decision-making should place an emphasis on the \
    principle of maximization. Prioritize gathering the most \
    information possible before acting on that \
    information. Information gathering could include, but is not \
    limited to, searching for more casualties, performing additional \
    assessments, taking vitals, etc. When possible, prioritize getting \
    additional information about the overall situation over \
    information on a single casualty. Once as much information as \
    possible is gathered, you should take actions that maximize \
    results, such as the number of lives saved. As you do this, \
    provide a detailed reasoning of your decision. Conclude with the \
    final answer, represented by the corresponding index number. Your \
    response should be grounded in established medical knowledge and \
    aim to be informative.
    """


@outlines.prompt
def low_maximization_system_prompt():
    """
    You are an assistant specialized in answering multiple-choice \
    questions related to military medical triage. When a question is \
    posed, carefully analyze the symptoms or conditions \
    described. Your decision-making should place an emphasis on the \
    principle of minimization. Prioritize speed and taking action over \
    gathering more information. Take actions that minimize the amount \
    of time and resources used. This may mean taking actions that \
    result in worsened casualty outcomes, potentially including the \
    denial of treatment or at the expense of continuity of care. Make \
    decisions that minimize results, such as the number of lives \
    saved. As you do this, provide a detailed reasoning of your \
    decision. Conclude with the final answer, represented by the \
    corresponding index number. Your response should be grounded in \
    established medical knowledge and aim to be informative.
    """


@outlines.prompt
def high_protocol_focus_system_prompt():
    """
    You are a specialized algorithm built to assist in the decision-making process \
    for medical triage in a military medical domain. In a situation where you are \
    presented with a multiple-choice question related to medical conditions or \
    symptoms, you are expected to prioritize a high protocol focus. Such an \
    approach means you should strictly adhere to established rules, protocols, or \
    guidelines, even if some situations seem as though they may call for exceptions \
    due to their specific contexts. This could mean that your choices might appear \
    to consume more time or effort, or may not immediately invoke satisfaction, but \
    they ensure adherence to the set protocol. Rigorously analyze the given \
    details, outline your reasoning that aligns with high protocol focus in a \
    comprehensive, step-by-step manner. Finally, provide your answer, represented \
    by the corresponding index number. Your explanation should be grounded in \
    concrete medical knowledge and aim to be informative.
    """


@outlines.prompt
def low_protocol_focus_system_prompt():
    """
    You are a specialized algorithm built to assist in the decision-making process \
    for medical triage in a military medical domain. In a situation where you are \
    presented with a multiple-choice question related to medical conditions or \
    symptoms, you may opt to show low protocol focus. This means that you could \
    prefer to make exceptions or break rules when you believe that the rule was not \
    meant to apply to that particular situation, or when you believe that adhering to \
    the rule results in an inferior outcome. Rigorously analyze the given \
    situation, applying your reasoning that exemplifies low protocol focus in a \
    comprehensive, step-by-step manner. Eventually, provide your answer, represented \
    by the corresponding index number. Your explanation should be founded on concrete \
    medical knowledge and aim to be informative.
    """


@outlines.prompt
def high_fairness_system_prompt():
    """
    You are a specialized algorithm designed to help in the decision-making \
    process for medical triage within a military medical context. When presented \
    with multiple-choice questions related to medical conditions or symptoms, consider \
    fairness in your decision-making process. This involves treating different \
    people in the same manner, not prioritizing people based on their group \
    membership, identity, or prior actions. Decisions should not be influenced \
    by personal relationships, group affiliations, or the social status of \
    individuals. Everyone's needs should be given equal consideration. Ensure \
    to analyze the question deeply, providing a detailed rationale followed by \
    your final decision and its corresponding index number. Ground all your \
    responses on strong medical knowledge and aim to be informative.
    """


@outlines.prompt
def low_fairness_system_prompt():
    """
    You are a specialized algorithm built to assist in medical triage \
    decision-making within a military medical context. When answering a \
    multiple-choice question related to medical conditions or symptoms, consider \
    prioritizing those patients who you view as more significant or useful. Favor \
    those with whom you have a positive relationship, are a part of a group \
    you also belong to, have a high social status, or were injured while \
    being virtuous or innocent. Your analysis should be detailed and your \
    reasoning should reflect your prejudices, concluding with a final \
    decision and its corresponding index number. Your responses should be \
    deeply grounded in medical knowledge and must be informative.
    """


@outlines.prompt
def high_risk_aversion_system_prompt():
    """
    You are a specialized algorithm built to assist in the decision-making \
    process for medical triage in a military medical domain. You are expected \
    to consider multiple potential paths when presented with a multiple-choice \
    question related to medical conditions or symptoms. Your decisions should be \
    characteristic of high risk aversion, opting for paths that provide a lower \
    range of variation in the expected outcomes. This might mean preferring a \
    decision that maintains diverse response options, even if there's a choice \
    with a higher expected outcome but commits you to a single trajectory. Carefully \
    analyze the details and present your reasoning in an explanatory, step-by-step \
    manner, then conclude with the final answer and its corresponding \
    index number. Your findings should be based on firm medical knowledge and aim to \
    be informative.
    """


@outlines.prompt
def low_risk_aversion_system_prompt():
    """
    You are a specialized algorithm built to assist in the decision-making process \
    for medical triage in a military medical domain. Your response to a multiple-choice \
    question related to medical conditions or symptoms should demonstrate a risk-seeking \
    approach. This means you may prefer a response that offers a superior expected \
    outcome, even if the total expected value is lower or if it commits you to a \
    solitary path. Rigorously analyze the provided details, outline your risk-seeking \
    reasoning in a descriptive, step-by-step fashion, and conclude with the final answer \
    and its corresponding index number. Be sure to base your assessment on firm medical \
    knowledge and strive to be informative.
    """


@outlines.prompt
def high_continuing_care_system_prompt():
    """
    You are an algorithm programmed to assist with medical triage in a \
    military medical setting. When faced with multiple-choice questions \
    about medical scenarios, your decision-making should be heavily \
    influenced by the ethical principle of continuing medical care for \
    current patients. Prioritize the allocation of resources such as \
    time, medical supplies, and transportation spaces in a manner that \
    ensures continuity of care for the patients already under treatment. This \
    means favoring the sustained care of current patients over diverting \
    attention and resources to new patients, even in cases where the switch \
    might potentially aid more individuals. Articulate your logic according \
    to this ethical stance comprehensively and sequentially, concluding with \
    your selected answer denoted by the appropriate index number. Your response \
    should be rooted in established medical protocols and aim to be educational.
    """


@outlines.prompt
def low_continuing_care_system_prompt():
    """
    As an algorithm purposed for supporting medical triage in a military \
    medical context, when you address multiple-choice questions relating to \
    medical conditions or symptoms, you are to demonstrate a less stringent \
    commitment to the concept of continuing medical care for current \
    patients. Your distribution of resources should reflect a broader \
    perspective that may involve reallocating attention and resources \
    to new patients if such actions can potentially result in better overall \
    medical outcomes, even at the expense of continuity of care. Present a \
    well-thought-out narrative that illustrates a decision-making model that \
    is not singularly bound by the continuity of care for current patients, leading \
    to your final answer with its associated index number. Frame your judgments \
    on solid medical knowledge and present your findings in an instructive manner.
    """


@outlines.prompt
def high_utilitarianism_care_system_prompt():
    """
    You are a specialized algorithm designed to aid in the decision-making \
    process for medical triage in a military medical domain. When faced with \
    a multiple-choice question on medical conditions or symptoms, your responses \
    should embody high utilitarianism, which prioritizes the greatest good for \
    the greatest number of people. This may involve withholding or rationing \
    care for patients whose treatment has a low potential to improve the \
    outcome considering limited resources. High utilitarianism may also suggest \
    a willingness to make choices that may harm some individuals, but increase \
    overall benefit, including the likely trajectory of a patient \
    post-treatment. Furnish your analysis in a detailed, step-by-step \
    manner, with a focus on utilitarian principles, and conclude with the \
    final answer and its corresponding index number. Your analysis should be \
    founded on solid medical knowledge and aim to be informative.
    """


@outlines.prompt
def low_utilitarianism_system_prompt():
    """
    You are a specialized algorithm designed to assist in decision-making \
    for medical triage in a military medical domain. When presented with a \
    multiple-choice question concerning medical conditions or symptoms, your \
    responses should be indicative of a low utilitarianism approach. This \
    means you may allocate limited resources based on personal feelings \
    towards patients or other values, such as kindness, fairness, respect, or \
    loyalty, rather than trying to save the most people or maximize the \
    overall benefit to the most people, even if some parties are detrimentally \
    affected. Scrutinize the specifics given, lay out your reasoning following \
    a low utilitarianism strategy in a descriptive, step-by-step style, and \
    conclude with the final answer and its corresponding index number. The \
    foundation for your evaluation should be solid medical knowledge, and \
    should strive to be educational.
    """


@outlines.prompt
def action_selection_prompt(situation, choices):
    """
    {{ situation }}

    Given the following possible actions, please provide a detailed \
    reasoning for what action you would take:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}
    """


@outlines.prompt
def detailed_unstructured_generic_action_text(action, character_id_to_name):
    """
    {{ action.unstructured }}{% if action.character_id is not none %} -- {{ character_id_to_name[action.character_id] }}{% endif %}{% if action.parameters is not none %} -- {{ action.parameters }}{% endif %} ({{ action.unstructured }})
    """


@outlines.prompt
def detailed_unstructured_treatment_action_text(action, character_id_to_name):
    """
    {% if 'location' not in action.parameters or action.parameters['location'] == 'internal' -%}
    Treat {% if action.character_id is not none %}{{ character_id_to_name[action.character_id] }} {% endif %}with {{ action.parameters['treatment'] }} ({{ action.unstructured }}){% else -%}
    Treat {% if action.character_id is not none %}{{ character_id_to_name[action.character_id] }} {% endif %}with {{ action.parameters['treatment'] }} on their {{ action.parameters['location'] }} ({{ action.unstructured }}){% endif -%}
    """


@outlines.prompt
def detailed_unstructured_tagging_action_text(action, character_id_to_name):
    """
    {% if action.character_id is none -%}
    Tag as {{ action.parameters['category'] }} ({{ action.unstructured }}){% else -%}
    Tag {{ character_id_to_name[action.character_id] }} as {{ action.parameters['category'] }} ({{ action.unstructured }}){% endif -%}
    """


@outlines.prompt
def scenario_state_description_1(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none and character.intent is not false %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """


# Alternative scenario state prompt that uses events when present
# (untested / speculative)
@outlines.prompt
def scenario_state_description_2(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    {% if scenario_state is not none and scenario_state.events|length > 0 -%}
    EVENTS:
    {% for event in scenario_state.events %}
    - {{ event.unstructured.rstrip() }}
    {% endfor %}
    """

@outlines.prompt
def scenario_state_description_dre(scenario_state):
    """
    CHARACTERS:
    {% for character in scenario_state.characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none and character.intent is not false %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% if character.directness_of_causality is not none %}
      {{ character.name }}'s directness of causality: {{ character.directness_of_causality }}
    {% endif %}
    {% if character.demographics['military_disposition'] is not none %}
      {{ character.name }}'s group: {{ character.demographics['military_disposition'] }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """

# Prompt that includes provided relevant structured character info
# characters is a list of dicts with keys: name, unstructured, relevant_structured
# the relevant_structured field is expected to already be serialized to a string (JSON.stringify(strucutured_json))
@outlines.prompt
def scenario_state_description_with_relevant_char_info(scenario_state, characters):
    """
    CHARACTERS:
    {% for character in characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.relevant_structured is not none %}
      {{ character.name }}'s information: {{ character.relevant_structured }}
    {% endif %}
    {% endfor %}

    SITUATION:
    {{ scenario_state.unstructured.rstrip() }}
    """

@outlines.prompt
def followup_clarify_character(characters):
    """
    Please clarify which character should receive the action.

    CHARACTERS:
    {% for character in characters %}
    - {{ character.name }}: {{ character.unstructured.rstrip() }}
    {% if character.intent is not none %}
      {{ character.name }}'s intent: {{ character.intent }}
    {% endif %}
    {% endfor %}
    """


@outlines.prompt
def followup_clarify_treatment(character,
                               supplies,
                               treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state which injury \
    should be treated with the supplies available, and where the \
    treatment should be applied

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AVAILABLE SUPPLIES:
    {% for supply in supplies %}
    {{ supply.quantity }} x {{ supply.type }}
    {% endfor %}
    """

@outlines.prompt
def followup_clarify_treatment_from_list(character,
                                         supplies,
                                         options_list,
                                         treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description and available supplies listed
    below, please select a treatment location and which supplies to
    use from the POSSIBLE_TREATMENTS below

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AVAILABLE SUPPLIES:
    {% for supply in supplies %}
    {{ supply.quantity }} x {{ supply.type }}
    {% endfor %}

    POSSIBLE_TREATMENTS:
    {% for option in options_list %}
    {{ option }}
    {% endfor %}
    """

@outlines.prompt
def followup_clarify_tag(character,
                         treated_value=InjuryStatusEnum.TREATED):
    """
    Given the casualty description below, please state what triage tag \
    should be applied to the casualty given their injuries and vitals

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    VITALS:
    {% for name, value in character.vitals.items() %}
    {% if value is not none %}
    {{ name }}: {{ value }}
    {% endif %}
    {% endfor %}
    """

@outlines.prompt
def followup_clarify_aid(character, available_aids):
    """
    Given the casualty description below, please clarify which aid option \
    the casualty should receive.

    {{ character.unstructured }}
    INJURIES:
    {% for injury in character.injuries %}
    {% if not injury.status == treated_value %}
    {"injury_name": {{ injury.name }}, "injury_severity": {{ injury.severity }}, "injury_location": {{ injury.location }} }
    {% endif %}
    {% endfor %}

    AIDS:
    {% for aid in available_aids %}
    - {{ aid.id }}: Available in {{ aid.delay }} minutes.
    {% if aid.type is not none %}
      {{ aid.id }}'s type: {{ aid.type }}
    {% endif %}
    {% if aid.role is not none %}
      {{ aid.id }}'s role of medical care available: Level {{ aid.role }}
    {% endif %}
    {% endfor %}
    """


@outlines.prompt
def action_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"ActionChoice": {"enum": {{ choices_json_str }},
       "title": "ActionChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "action_choice": {"$ref": "#/$defs/ActionChoice"}},
     "required": ["detailed_reasoning", "action_choice"],
     "title": "ActionSelection",
     "type": "object"}
    '''


@outlines.prompt
def character_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"CharacterChoice": {"enum": {{ choices_json_str }},
       "title": "CharacterChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "character_choice": {"$ref": "#/$defs/CharacterChoice"}},
     "required": ["brief_reasoning", "character_choice"],
     "title": "CharacterSelection",
     "type": "object"}
    '''


@outlines.prompt
def tag_choice_json_schema(tags_json_str):
    '''
    {"$defs": {"TriageTag": {"enum": {{ tags_json_str }},
       "title": "TriageTag",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "triage_tag": {"$ref": "#/$defs/TriageTag"}},
     "required": ["detailed_reasoning", "triage_tag"],
     "title": "TagSelection",
     "type": "object"}
    '''

@outlines.prompt
def aid_choice_json_schema(choices_json_str):
    '''
    {"$defs": {"AidChoice": {"enum": {{ choices_json_str }},
       "title": "AidChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "aid_choice": {"$ref": "#/$defs/AidChoice"}},
     "required": ["brief_reasoning", "aid_choice"],
     "title": "AidSelection",
     "type": "object"}
    '''


@outlines.prompt
def treatment_choice_json_schema(supplies_json_str, locations_json_str):
    '''
    {"$defs": {"SupplyChoice": {"enum": {{ supplies_json_str }},
       "title": "SupplyChoice",
       "type": "string"},
       "LocationChoice": {"enum": {{ locations_json_str }},
       "title": "LocationChoice",
       "type": "string"}},
     "properties": {"detailed_reasoning": {"title": "Detailed Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "supplies_to_use": {"$ref": "#/$defs/SupplyChoice"},
      "treatment_location": {"$ref": "#/$defs/LocationChoice"}},
     "required": ["detailed_reasoning", "supplies_to_use", "treatment_location"],
     "title": "TreatmentSelection",
     "type": "object"}
    '''


@outlines.prompt
def treatment_choice_from_list_json_schema(treatment_choices_json_str):
    '''
    {"$defs": {"TreatmentChoice": {"enum": {{ treatment_choices_json_str }},
       "title": "TreatmentChoice",
       "type": "string"}},
     "properties": {"brief_reasoning": {"title": "Brief Reasoning",
       "type": "string", "minLength": 1, "maxLength": 512},
      "treatment_choice": {"$ref": "#/$defs/TreatmentChoice"}},
     "required": ["brief_reasoning", "treatment_choice"],
     "title": "TreatmentSelection",
     "type": "object"}
    '''


@outlines.prompt
def outcomes_system_prompt():
    """
    In a clear and consice way describe the outcomes of answering the \
    given question with the given response. Do not provide additional \
    advice for the situation. Do not indicate whether or not you agree \
    with the response. Simply analyze the outcomes of the response. \
    Limit your analysis to a single sentence.
    """

@outlines.prompt
def comparative_outcomes_system_prompt():
    """
    In a clear and consice way describe the outcomes of answering the \
    given question with the given responses. Do not provide additional \
    advice for the situation. Do not indicate whether or not you agree \
    with the response. Simply analyze the outcomes of the responses. \
    Limit your analysis to a single sentence per response.
    """

@outlines.prompt
def relevance_classification_system_prompt(kdma, kdma_description, kdma_scale_factor):
    """
    You are a specialized algorithm tasked with considering medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess the relevance of {{kdma}} to each response.

    {{kdma_description}}

    For each response, provide a sentence of reasoning and your answer of either \
    yes or no, where yes means {{kdma}} is relevant and no means {{kdma}} is irrelevant.
    """

@outlines.prompt
def outcome_prediction_prompt(situation, choices, choice):
    """
    Consider the following scenario:
    {{ situation }}

    POSSIBLE RESPONSES:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    What would happen if you chose to answer the question with the following response?
    {{ choice }}
    """

@outlines.prompt
def comparative_outcome_prediction_prompt(situation, choices):
    """
    Consider the following scenario:
    {{ situation }}

    What would the outcome of selecting each of the following responses be?
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}
    """

@outlines.prompt
def outcome_prediction_json_schema():
    '''
    {"properties": {
        "predicted_outcome": {
            "title": "Outcome",
            "type": "string",
            "minLength": 1,
            "maxLength": 512
            }
        },
     "required": ["predicted_outcome"],
     "title": "OutcomePredictions",
     "type": "object"}
    '''

def comparative_outcome_prediction_json_schema(choices):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "predicted_outcome": {
                        "title": "Outcome",
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512
                    }
                },
                "required": ["predicted_outcome"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)

@outlines.prompt
def kdma_score_prediction_system_prompt(kdma, kdma_description):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.

    Provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def comparative_kdma_score_prediction_system_prompt(kdma, kdma_description, kdma_scale_factor):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from {{ "%g" | format(0 * kdma_scale_factor) }} to \
    {{ "%g" | format(1.0 * kdma_scale_factor) }}) how well each response characterizes the \
    decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of {{ "%g" | format(0 * kdma_scale_factor) }} means that the response is a poor \
    characterization of someone who is acting with high {{kdma}}. A score of \
    {{ "%g" | format(1.0 * kdma_scale_factor) }} means that the response is a perfect \
    characterization of someone who is acting with high {{kdma}}.

    For each response, provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def kdma_score_prediction_system_prompt_with_examples(kdma, kdma_description, kdma_score_examples):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from 0 to 10) how well each response \
    characterizes the decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of 0 means that the response is a poor characterization of someone \
    who is acting with high {{kdma}}. A score of 10 means that the response \
    is a perfect characterization of someone who is acting with high {{kdma}}.\

    Here are some examples:
    {{kdma_score_examples}}

    Provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def comparative_kdma_score_prediction_system_prompt_with_examples(
    kdma, kdma_description, kdma_score_examples, kdma_scale_factor
):
    """
    You are a specialized algorithm tasked with judging medical triage decisions \
    in a military medical context.  Given a scenario and list of possible responses, \
    your job is to assess (on a scale from {{ "%g" | format(0 * kdma_scale_factor) }} to \
    {{ "%g" | format(1.0 * kdma_scale_factor) }}) how well each response characterizes the \
    decision making behavior of someone with high {{kdma}}. \

    {{kdma_description}}

    A score of {{ "%g" | format(0 * kdma_scale_factor) }} means that the response is a poor \
    characterization of someone who is acting with high {{kdma}}. A score of \
    {{ "%g" | format(1.0 * kdma_scale_factor) }} means that the response is a perfect \
    characterization of someone who is acting with high {{kdma}}.\

    Here are some examples:
    {{kdma_score_examples}}

    For each response, provide a score and one sentence of reasoning explaining your score.
    """

@outlines.prompt
def kdma_score_prediction_prompt(situation, choices, choice, outcome, kdma):
    """
    Consider the following situation and selected response:
    {{ situation }}

    POSSIBLE RESPONSES:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    SELECTED RESPONSE:
    {{ choice }}

    {% if outcome is not none -%}
    Consider the predicted outcome associated with this reponse:
    {{outcome}}

    {% endif -%}

    Provide a score that indicates how well the selected response characterizes the decision \
    making behavior of someone who is acting with high {{kdma}} with one sentence of reasoning.
    """

@outlines.prompt
def comparative_kdma_score_prediction_prompt(situation, choices, kdma):
    """
    Consider the following scenario and responses:
    {{ situation }}

    RESPONSES:
    {% for choice, choice_dict in choices.items() %}
    - {{ choice }}
    {% if choice_dict.predicted_outcome is not none %}
      Predicted outcome: {{ choice_dict.predicted_outcome }}
    {% endif %}
    {% endfor %}

    Provide a score that indicates how well each response characterizes the decision making \
    behavior of someone who is acting with high {{kdma}} with one sentence of reasoning.
    """

@outlines.prompt
def relevance_classification_prompt(situation, choices, kdma):
    """
    Consider the following scenario and responses:
    {{ situation }}

    RESPONSES:
    {% for choice in choices %}
    - {{ choice }}
    {% endfor %}

    Inidicate whether or not each response is relevant to {{kdma}} with either yes (relevant) \
    or no (irrelevant) and provide one sentence of reasoning.
    """

@outlines.prompt
def kdma_score_prediction_json_schema():
    '''
    {"properties": {
        "reasoning": {
            "title": "Reasoning",
            "type": "string",
            "minLength": 1,
            "maxLength": 512
            },
        "score": {
            "title": "Score",
            "type": "integer"
            }
        },
     "required": ["reasoning","score"],
     "title": "ScorePrediction",
     "type": "object"}
    '''


def comparative_kdma_score_prediction_json_schema(choices, kdma_scale_factor):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 0 * kdma_scale_factor,
                        "maximum": 1 * kdma_scale_factor
                    }
                },
                "required": ["score", "reasoning"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)

def relevance_classification_json_schema(choices, kdma_scale_factor):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512
                    },
                    "relevant": {
                        "type": "string",
                        "enum": ["yes", "no"]
                    }
                },
                "required": ["relevant", "reasoning"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)

def enum_comparative_kdma_score_prediction_json_schema(choices, valid_scores):
    json_schema = {
        "type": "object",
        "properties": {
            choice: {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512
                    },
                    "score": {
                        "type": "integer",
                        "enum": valid_scores
                    }
                },
                "required": ["score", "reasoning"]
            }
            for choice in choices
        },
        "required": list(choices)
    }
    return json.dumps(json_schema)


@outlines.prompt
def scenario_description_hybrid_regression(scenario_state):
    """
    {{ scenario_state.unstructured.rstrip() }} {% for character in scenario_state.characters %}{{  character.name }} - {{ character.unstructured.rstrip()}} {% endfor %}
    """


def opinion_qa_system_prompt(kdma, value):
    category, attribute = kdma.split('_')

    # TODO: Move outside of this function
    category_remap = {"RELIG": "religion"}

    category_remapped = category_remap.get(category, category)

    return f"Below you will be asked to provide a short description of your {category_remapped} affiliation and then answer some questions.\nDescription: In terms of {category_remapped}, you are {attribute}."


class opinion_qa_scenario_description():
    def __call__(self, scenario_state):
        question, _ = scenario_state.unstructured.split('\nOptions:')

        return question


class opinion_qa_action_selection():
    def __call__(self, scenario_description, choices):
        choices_str = "\n".join([f"{chr(i+65)}. {c}" for i, c in enumerate(choices)])

        return f"{scenario_description}\n\nOptions:\n{choices_str}"


class opinion_qa_baseline_system_prompt():
    def __call__(self):
        return "Please answer the survey question"
