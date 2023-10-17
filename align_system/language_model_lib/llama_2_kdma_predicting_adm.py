import json
from typing import Union, List, Dict, Tuple, Optional, TextIO
from align_system.language_model_lib.chat_language_model import ChatLanguageModel
from align_system.language_model_lib.util import extract_kdma_description

class Llama2KDMAPredictingADM(ChatLanguageModel):
    
    def predict_outcomes(self,
                         scenario_text: str,
                         probe_text: str,
                         choices: List[str],
                         log_file: Optional[TextIO] = None,
                         max_tokens: int = 512,
                         temperature: float = 0.6,
                         outcome_template_file: str = 'templates/predict_outcomes.md') -> List[str]:
        """
        Predicts outcomes for given scenario, probe and choices.

        :param scenario: Scenario text.
        :param probe: Probe text.
        :param choices: Choices text.
        :param log_file: Optional log file.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Temperature for sampling.
        :param outcome_template_file: Template file for Outcomes.
        :return: List of generated predictions.
        """
        return self.generate_from_template(
            outcome_template_file,
            [
                {
                    'scenario': scenario_text,
                    'probe': probe_text, 
                    'choice': choice,
                }
                for choice in choices 
            ],
            log_file=log_file,
            max_tokens=max_tokens,
            temperature=temperature
        )

    
    def predict_kdma_scores(self,
                            scenario_text: str,
                            probe_text: str,
                            choice_texts: List[str],
                            predicted_outcomes: Optional[List[str]] = None,
                            generate_reasoning: bool = True,
                            log_file: Optional[TextIO] = None,
                            max_new_tokens: int = 512,
                            temperature: float = 0.6,
                            kdma_template_file: str = 'templates/kdma.md',
                            kdma_descriptions_file: str = 'templates/bbn_kdma_descriptions.md') -> Union[List[Dict[str, float]], Tuple[List[Dict[str, float]], List[Dict[str, str]]]]:
        """
        Predicts KDMA scores each choice text under the given scenario and probe.

        :param scenario_text: Scenario text.
        :param probe_text: Probe text.
        :param choice_texts: Choices text.
        :param predicted_outcomes: Predicted outcomes.
        :param generate_reasoning: Flag to generate reasoning.
        :param log_file: Optional log file.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Temperature for sampling.
        :param kdma_template_file: Template file for KDMA prediction.
        :param kdma_descriptions_file: Template file for KDMA descriptions.
        :return: KDMA predictions. If generate_reasoning is True, return predictions and reasonings.
        """
        choice_ids = [f'choice_{i}' for i in range(len(choice_texts))]
        substitutions = []
        info = []
        kdma_descriptions = extract_kdma_description(kdma_descriptions_file)
        if predicted_outcomes is None:
            predicted_outcomes = [None] * len(choice_texts)
        
        for choice_id, choice, outcome in zip(choice_ids, choice_texts, predicted_outcomes):
            for kdma, kdma_description in kdma_descriptions.items():
                substitution = {
                    'kdma': kdma,
                    'kdma_description': kdma_description,
                    'scenario': scenario_text,
                    'probe': probe_text,
                    'choice': choice,
                }
                
                if outcome is not None:
                    substitution['outcome'] = outcome
                    
                substitutions.append(substitution)
                info.append((choice_id, kdma))
        
        def parse_kdma_score_response(response: str) -> Dict[str, Union[float, str]]:
            """
            Parses KDMA score response.

            :param response: Response to parse.
            :return: Dictionary with KDMA score and reasoning if generate_reasoning.
            """
            if generate_reasoning:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                response_json = json.loads(response[start_idx:end_idx+1])
                assert 'score' in response_json, 'score not found in response'
                assert 'reasoning' in response_json, 'reasoning not found in response'
            else:
                # find the first numeric character
                char = None
                for c in response:
                    if c.isnumeric():
                        char = c
                        break
                assert char is not None, 'Could not find numeric character in response'
                response_json = {
                    'score': float(response[response.find(char):])
                }                
            return response_json
            
        generations = self.generate_from_template(
            kdma_template_file,
            substitutions,
            parse_kdma_score_response,
            log_file=log_file,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        predicted_kdmas = {}
        reasonings = {}
        for (choice_id, kdma), generation in zip(info, generations):
            predicted_choice_kdmas = predicted_kdmas.get(choice_id, {})
            predicted_kdmas[choice_id] = predicted_choice_kdmas
            
            choice_reasonings = reasonings.get(choice_id, {})
            reasonings[choice_id] = choice_reasonings
            
            predicted_choice_kdmas[kdma] = generation['score']
            
            if generate_reasoning:
                choice_reasonings[kdma] = generation['reasoning']
        
        predicted_kdmas = [
            predicted_kdmas[choice_id]
            for choice_id in choice_ids
        ]
        if generate_reasoning:
            reasonings = [
                reasonings[choice_id]
                for choice_id in choice_ids
            ]
        
        if generate_reasoning:
            return predicted_kdmas, reasonings
        else:
            return predicted_kdmas