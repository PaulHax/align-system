from align_system.algorithms.abstracts import AlignedDecisionMaker
from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel


def convert_to_string(data):
    # Mapping for full names
    full_names = {
        "HR_bpm": "Heart Rate",
        "RR_bpm": "Respiration Rate",
        "SBP_mmHg": "Systolic Blood Pressure",
        "CoreTemp_C": "Core Body Temperature",
        "PPI": "Pulse Pressure Index",
        "BO2PP_mmHg": "Blood Oxygen Partial Pressure",
        "GCS": "Glasgow Coma Scale",
        "SkinColor": "Skin Color",
        "SkinTouch": "Skin Touch",
        "CRT_s": "Capillary Refill Time",
        "RadialPulsePalpability": "Radial Pulse Palpability"
    }

    # Convert and round the values
    result = []
    for key, value in data.items():
        name = full_names.get(key, key)
        if isinstance(value, float):
            value = round(value, 3)  # Adjust the rounding as needed
        result.append(f"{name}: {value}")

    return "\n".join(result)

class PulseTaggingADM(ChatLanguageModel, AlignedDecisionMaker):


    def identify_tag_color(self, chat_response):
        # Limit parsing tags mentioned before conclusion
        if '"Answer"' in chat_response:
            chat_response = chat_response[chat_response.find('"Answer"'):]

        for i, color in enumerate(['GREEN', 'YELLOW', 'RED', 'BLACK']):
            if color in chat_response.upper():
                return i

        for i, color in enumerate(['MINOR', 'DELAYED', 'IMMEDIATE', 'EXPECTANT']):
            if color in chat_response.upper():
                return i

        return -1

    def __call__(self, sample, target_kdma_values, **kwargs):
        # The start_guide is a text formatting of the START guide flowchart
        # TODO move these strings to template filess
        start_guide = '''**START Adult Triage Guide for Tagging Casualties**

1. **Initial Assessment: Ability to Walk**
   - If it is unknown whether the patient is able to walk proceed to the next check.
   - If the patient is able to walk, they are tagged as **MINOR** (Green Triage Tag Color) and directed to secondary triage. Otherwise, proceed to the next check.

2. **Spontaneous Breathing Check**
   - If the patient is breathing after the airway is positioned, move to the next step. Otherwise, tag them as *EXPECTANT** (Black Triage Tag Color).

3. **Respiratory Rate Check**
   - If the respiratory rate is greater than or equal to 30 breaths per minute, tag the casualty as **IMMEDIATE** (Red Triage Tag Color). Otherwise, proceed to the next check.

4. **Perfusion Check**
   - Assess the radial pulse or capillary refill time. If the radial pulse is absent or capillary refill time is greater than 2 seconds, tag the casualty as **IMMEDIATE** (Red Triage Tag Color). Otherwise, proceed to the next check.

5. **Mental Status Check**
   - If the patient does not obey commands, tag them as **IMMEDIATE** (Red Triage Tag Color). Otherwise, tag them as **DELAYED** (Yellow Triage Tag Color).

**Additional Information on Triage Categories:**
- **EXPECTANT:** Victim unlikely to survive. Provide palliative care.
- **IMMEDIATE:** Victim needs immediate help. Includes medical attention within minutes for survival.
- **DELAYED:** Serious injuries, but not life-threatening and not expected to deteriorate soon.
- **MINOR:** Minor injuries. May be able to assist in their own care.
'''
        sample_1 = 'Patient is able to walk to the designated safety area.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is able to walk, so they can be tagged as MINOR (Green). It is not neccessary to perform the other checks.

Conclusion: Tag the casualty as MINOR (Green).'''

        sample_2 = 'The patient is conscious. Patient has a moderate hemorrhage. You have used a tourniquet to stop the bleeding. Patient is breathing normally. Patient pulse is within the normal range. The patient is able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

It is unclear if the patient is able to walk, so I will move on to the next check.

2. Spontaneous Breathing Check

The patient is breathing, so I will proceed to the next check.

3. Respiratory Rate Check

The patient is breathing normally, which is less than 30 breaths per minute, so I should proceed to the next check.

4. Perfusion Check

The patient's pulse is within the normal range, so I will move to next check.

5. Mental Status Check

The casualty follows basic commands, so they should be tagged as DELAYED (Yellow).

Conclusion: Tag the casualty as DELAYED (Yellow)'''

        sample_3 = 'The patient is conscious. Patient is breathing normally. Patient pulse is abnormally high. Capillary refill time is more than 3 seconds. The patient is not able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is not able to walk, so I will move on to the next check.

2. Spontaneous Breathing Check

The patient is breathing, so I should move to the next check.

3. Respiratory Rate Check

The patient is breathing, but their respiratory rate is not faster than 30 breaths per minute, so I must proceed to the next check to get more information.

4. Perfusion Check

The patient's pulse is abnormally high and capillary refill time is more than 3 seconds, so they should be tagged as IMMEDIATE (Red). I've determined the appropriate tag color, so I don't need to perform the next check.

Conclusion: Tag the casualty as IMMEDIATE (Red)'''

        sample_4 = 'The patient is unconscious. Patient is not breathing. Repositioning the airway did not help. Patient has a moderate hemorrhage. You have used a tourniquet to stop the bleeding. No radial pulse can be detected on the patient. Capillary refill time is more than 3 seconds. The patient is not able to follow basic commands.', '''Let's apply the START guide

1. Initial Assessment: Ability to Walk

The patient is unconscious and not able to walk, so I must continue on to the next check.

2. Spontaneous Breathing Check

The patient is not breathing so this patient should be tagged as EXPECTANT (Black). No further checks are needed to determine how this patient should be tagged.

Conclusion: Tag the casualty as EXPECTANT (Black)'''

        dialogs = [[
            {
                'role': 'system',
                'content': start_guide
            },
            {
                'role': 'user',
                'content': sample_3[0]
            },
            {
                'role': 'assistant',
                'content': sample_3[1]
            },
            {
                'role': 'user',
                'content': sample_1[0]
            },
            {
                'role': 'assistant',
                'content': sample_1[1]
            },
            {
                'role': 'user',
                'content': sample_2[0]
            },
            {
                'role': 'assistant',
                'content': sample_2[1]
            },
            {
                'role': 'user',
                'content': sample_4[0]
            },
            {
                'role': 'assistant',
                'content': sample_4[1]
            },
            {
                'role': 'user',
                'content': sample['scenario']
            },
            {
                'role': 'assistant',
                'content': 'Let\'s apply the START guide'
            }
        ]]



        n_samples = kwargs.get('n_samples', 1)  # Set the number of samples, default to 5 if not provided

        chat_responses = []  # List to store the generated responses


        chat_responses = self.generate_responses(dialogs * n_samples)
            # chat_responses.append(response)

        color_counts = {}  # Dictionary to store the count of each color_idx

        for response in chat_responses:
            color_idx = self.identify_tag_color(response)
            color_counts[color_idx] = color_counts.get(color_idx, 0) + 1

        most_popular_color_idx = max(color_counts, key=color_counts.get)  # Get the most popular color_idx

        most_popular_reasoning = None

        for response in chat_responses:
            color_idx = self.identify_tag_color(response)
            if color_idx == most_popular_color_idx:
                most_popular_reasoning = response
                break

        return {
            'choice': most_popular_color_idx,
            'info': {
                'generated_reasoning': most_popular_reasoning,
                'color_counts': color_counts,
                'all_reasonings': chat_responses,
            }
        }
