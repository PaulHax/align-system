from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms.llm_chat_baseline import (
    LLMChatBaseline,
    TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT,
    TAGGING_MULTIPLE_CHOICE_JSON_FORMAT)
from align_system.prompt_engineering.common import (
    prepare_action_based_prompt,
    prepare_treatment_selection_prompt,
    prepare_tagging_selection_prompt)


def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    parser.add_argument('-m', '--model',
                        type=str,
                        help="Example command-line argument",
                        default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('-r', '--precision',
                        type=str,
                        help="Precision, must be 'full' or 'half' "
                             "(default: 'full')",
                        default='full')
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")
    parser.add_argument('--answer-attempts',
                        type=int,
                        default=5,
                        help="Max number of attempts at generating a "
                             "parseable response (default: 5)")


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    run_action_based_chat_system(
        **build_interfaces(
            add_cli_args, "ALIGN Action Based System CLI - Chat Model",
            supported_interfaces={'TA3ActionBased'}))


CASUALTY_TAGS = ['MINIMAL',
                 'DELAYED',
                 'IMMEDIATE',
                 'EXPECTANT']


def run_action_based_chat_system(interface,
                                 model,
                                 precision,
                                 align_to_target=False,
                                 answer_attempts=5):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target_dict = scenario.get_alignment_target()

    print('Creating algorithm')
    algorithm = LLMChatBaseline(hf_model=model, precision=precision)
    algorithm.load_model()

    current_state = scenario.get_state()
    scenario_complete = current_state.get('scenario_complete', False)

    while not scenario_complete:
        available_actions = scenario.get_available_actions()

        untagged_casualties = [c for c in current_state['casualties']
                               if 'tag' not in c]

        # Don't let ADM choose to tag a casualty unless there are
        # still untagged casualties
        available_actions_filtered =\
            [a for a in available_actions
             if a['action_type'] != 'TAG_CASUALTY'
             or (a['action_type'] == 'TAG_CASUALTY'
                 and len(untagged_casualties) > 0)]

        prompt = prepare_action_based_prompt(
            scenario_dict['state']['unstructured'],
            current_state['mission'].get('unstructured'),
            current_state['unstructured'],
            current_state['casualties'],
            available_actions=None,  # Available actions passed in later
            alignment_target=alignment_target_dict if align_to_target else None
        )
        print("* Prompt for ADM: {}".format(prompt))

        if len(available_actions_filtered) == 0:
            raise RuntimeError("No available actions from filtered list!")
        elif len(available_actions_filtered) == 1:
            print("** Choosing only available (filtered) action")
            action_idx = 0
        else:
            # TODO: More elegant failure case if we don't find an answer
            # within answer_attempts; currently just passing along bad
            # values
            for _ in range(answer_attempts):
                # TODO a possible improvement would be to use a separate
                # prompt to parse mis-formatted JSON instead of simply
                # trying again
                raw_response = algorithm.answer_multiple_choice(
                    prompt,
                    [a['unstructured'] for a in available_actions_filtered])

                print("* ADM raw response: {}".format(raw_response))

                parsed_output = LLMChatBaseline.attempt_generic_parse(
                    raw_response, ['Reasoning', 'Answer'])

                if parsed_output is None:
                    explanation, action_idx =\
                        LLMChatBaseline.parse_generated_output(raw_response)
                else:
                    explanation = parsed_output['Reasoning']
                    action_idx = parsed_output['Answer']

                if(explanation is not None
                   and action_idx is not None):
                    if len(available_actions_filtered) > action_idx:
                        break
                    else:
                        print('** Selected action_idx out of range of '
                              'available actions, retrying!')
                        continue

                print('** Failed to parse')

        action_to_take = available_actions_filtered[int(action_idx)]

        if explanation is not None:
            action_to_take['justification'] = explanation

        if action_to_take['action_type'] == 'APPLY_TREATMENT':
            # Ask the system to specify the treatment to use and where

            # First casualty with the matching ID (should only be one)
            casualty_id = action_to_take['casualty_id']
            matching_casualties = [c for c in current_state['casualties']
                                   if c['id'] == casualty_id]

            assert len(matching_casualties) == 1
            casualty_to_treat = matching_casualties[0]

            treatment_prompt = prepare_treatment_selection_prompt(
                casualty_to_treat['unstructured'],
                casualty_to_treat['vitals'],
                current_state['supplies'])

            print("** Treatment prompt for ADM: {}".format(treatment_prompt))

            for _ in range(answer_attempts):
                raw_treatment_response = algorithm.answer_multiple_choice(
                    prompt, [s['type'] for s in current_state['supplies']],
                    json_format=TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT)

                print("** ADM raw treatment response: {}".format(
                    raw_treatment_response))

                parsed_treatment_output = LLMChatBaseline.attempt_generic_parse(  # noqa
                    raw_treatment_response, ['Reasoning', 'Answer', 'Location'])  # noqa

                if parsed_treatment_output is not None:
                    treatment_idx = parsed_treatment_output['Answer']

                    if len(current_state['supplies']) <= treatment_idx:
                        print('** Selected treatment_idx out of range of '
                              'available treatment options, retrying!')
                        continue

                    treatment = current_state['supplies'][treatment_idx]['type']  # noqa

                    treatment_location = parsed_treatment_output['Location']

                    action_to_take['parameters'] = {
                        'treatment': treatment,
                        'location': treatment_location}

                    break
                else:
                    print('** Failed to parse treatment')
        elif action_to_take['action_type'] == 'TAG_CASUALTY':
            # Ask the system to specify which triage tag to apply

            tagging_prompt = prepare_tagging_selection_prompt(
                untagged_casualties,
                CASUALTY_TAGS)

            print("** Tagging prompt for ADM: {}".format(tagging_prompt))

            for _ in range(answer_attempts):
                raw_tagging_response = algorithm.answer_multiple_choice(
                    prompt, [c['unstructured'].strip()
                             for c in untagged_casualties],
                    json_format=TAGGING_MULTIPLE_CHOICE_JSON_FORMAT)

                print("** ADM raw tagging response: {}".format(
                    raw_tagging_response))

                parsed_tagging_output = LLMChatBaseline.attempt_generic_parse(  # noqa
                    raw_tagging_response, ['Reasoning', 'Answer', 'Tag'])  # noqa

                if parsed_tagging_output is not None:
                    casualty_idx = parsed_tagging_output['Answer']

                    if len(untagged_casualties) <= casualty_idx:
                        print('** Selected casualty_idx out of range of '
                              'available treatment options, retrying!')
                        continue

                    casualty_to_tag_id = untagged_casualties[casualty_idx]['id']  # noqa

                    tag = parsed_tagging_output['Tag']

                    # Populate required parameters for tagging action
                    action_to_take['casualty_id'] = casualty_to_tag_id
                    action_to_take['parameters'] = {'category': tag}

                    break
                else:
                    print('** Failed to parse tagging')

        current_state = scenario.take_action(action_to_take)

        scenario_complete = current_state.get('scenario_complete', False)


if __name__ == "__main__":
    main()
