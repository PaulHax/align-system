from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker
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
    
    
    def identify_tag_color(self, message):
        chat_response = self.generate_responses(
            dialogs=[[
                {
                    'role': 'system',
                    'content': 'Format the user\'s message as a JSON object identifying the tag color. GREEN, YELLOW, RED, BLACK'
                },
                # black tag example
                {
                    'role': 'user',
                    'content': 'Because the casualty is not breathing, they should be tagged as **EXPECTANT** (Black Triage Tag Color), indicating that they are unlikely to survive given the severity of injuries, level of available care, or both. Palliative care and pain relief should be provided.'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "BLACK" }'
                },
                # green tag example
                {
                    'role': 'user',
                    'content': 'They should be tagged GREEN'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "GREEN" }'
                },
                # yellow tag example
                {
                    'role': 'user',
                    'content': 'Due to the severity of the casualty\'s injuries, they should be tagged YELLOW'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "YELLOW" }'
                },
                # red tag example
                {
                    'role': 'user',
                    'content': 'The casualty is not breathing, but has a pulse. They should be tagged RED'
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color": "RED" }'
                },
                # actual example
                {
                    'role': 'user',
                    'content': message
                },
                {
                    'role': 'assistant',
                    'content': '{"tag_color":'
                }
            ]]
        )[0]
        for i, color in enumerate(['GREEN', 'YELLOW', 'RED', 'BLACK']):
            if color in chat_response.upper():
                return i
            
        for i, color in enumerate(['MINOR', 'DELAYED', 'IMMEDIATE', 'EXPECTANT']):
            if color in chat_response.upper():
                return i
        
        return -1
    
    def __call__(self, sample, target_kdma_values, **kwargs):
        start_guide = '''**START Adult Triage Guide for Tagging Casualties**

1. **Initial Assessment: Ability to Walk**
   - If the casualty is able to walk, they are tagged as **MINOR** (Green Triage Tag Color) and directed to secondary triage.

2. **Spontaneous Breathing Check**
   - If the casualty is not breathing even after the airway is positioned, they are tagged as **EXPECTANT** (Black Triage Tag Color), indicating they are unlikely to survive given the severity of injuries, level of available care, or both. Palliative care and pain relief should be provided.
   - If the casualty resumes spontaneous breathing after positioning the airway, proceed to the next check.

3. **Respiratory Rate Check**
   - If the respiratory rate is greater than or equal to 30 breaths per minute, tag the casualty as **IMMEDIATE** (Red Triage Tag Color), indicating the need for immediate intervention and transport.

4. **Perfusion Check**
   - Assess the radial pulse or capillary refill time. If the radial pulse is absent or capillary refill time is greater than 2 seconds, tag the casualty as **IMMEDIATE**.
   - If the radial pulse is present or capillary refill time is less than 2 seconds, proceed to the next check.

5. **Mental Status Check**
   - If the casualty does not obey commands, tag them as **IMMEDIATE**.
   - If the casualty obeys commands, tag them as **DELAYED** (Yellow Triage Tag Color), indicating that their transport can be delayed and they have serious but potentially not life-threatening injuries, and are not expected to deteriorate significantly over several hours.

**Additional Information on Triage Categories:**
- **EXPECTANT:** Victim unlikely to survive. Provide palliative care.
- **IMMEDIATE:** Victim needs immediate help. Includes medical attention within minutes for survival.
- **DELAYED:** Serious injuries, but not life-threatening and not expected to deteriorate soon.
- **MINOR:** Minor injuries. May be able to assist in their own care.
'''
        sample_1 = '''Casualty vitals:
Heart Rate: 72
Respiration Rate: 12
Systolic Blood Pressure: 114
Core Body Temperature: 37
Pulse Pressure Index: 0.07
Blood Oxygen Partial Pressure: 40
Glasgow Coma Scale: 15
Skin Color: normal
Skin Touch: normal
Capillary Refill Time: 1
Radial Pulse Palpability: true''', '''Let's follow the START triage guide:

1. Initial Assessment: Ability to Walk

Because the casualty has a systolic blood pressure greater than 108 they may be able to walk.

2. Spontaneous Breathing Check

The casualty is breathing faster than 5.5 times a minute, so they pass this check.

3. Respiratory Rate Check

The casualty's respiratory rate is 12, which is less than 30, so they pass this check.

4. Perfusion Check

The casualty's capillary refill time is 1, which is normal, so they pass this check.

5. Mental Status Check

The casualty's Glasgow Coma Scale is 15, which indicates a mild brain injury. Becuase the Glasgow Coma Scale is above 11, they should be tagged MINOR (GREEN).

Based on the assessment, the casualty is tagged as MINOR (GREEN).'''

        sample_2 = '''Heart Rate: 75.203
Respiration Rate: 11.858
Systolic Blood Pressure: 111.791
Core Body Temperature: 37.058
Pulse Pressure Index: 0.038
Blood Oxygen Partial Pressure: 38.643
Glasgow Coma Scale: 9
Skin Color: normal
Skin Touch: normal
Capillary Refill Time: 1
Radial Pulse Palpability: True''', '''Let's follow the START triage guide:

1. Initial Assessment: Ability to Walk

Because the casualty has a systolic blood pressure greater than 108 they may be able to walk

2. Spontaneous Breathing Check

The casualty is breathing faster than 5.5 times a minute, so they pass this check.

3. Respiratory Rate Check

The casualty's respiratory rate is 11.858, which is less than 30, so they pass this check.

4. Perfusion Check

The casualty's capillary refill time is 1, which is normal, so they pass this check.

5. Mental Status Check

The casualty's Glasgow Coma Scale is under 11, which indicates a mild to moderate brain injury. suggesting either a YELLOW or RED tag. Becuase the Glasgow Coma Scale is above 6.5, they should be tagged DELAYED (YELLOW).

Based on the assessment, the casualty is tagged as DELAYED (YELLOW)'''

        chat_response = self.generate_responses(
            dialogs = [[
                {
                    'role': 'system',
                    'content': start_guide
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
                    'content': convert_to_string(sample['scenario'])
                }
            ]]
        )

        color_idx = self.identify_tag_color(chat_response[0])
        
        return {
            'choice': color_idx,
            'info': {
                'generated_reasoning': chat_response[0]
            }
        }