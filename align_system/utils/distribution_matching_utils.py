from align_system.utils import kde_utils


def match_to_target_kde_sample(alignment_target, available_actions, kde_norm):
	'''
	Samples a random value from the target KDE and selects the action with KDMA value closest to the sample
	'''
	target_kdma = alignment_target.kdma_values[0] # TODO extend this to multi-KDMAs
	target_kde = kde_utils.load_kde(target_kdma, kde_norm)
	sampled_target = target_kde.sample(1)[0]
	
	selected_action = None
	min_dist = float('inf')
	for action in available_actions:
		if action.kdma_association is not None:
			choice_value = action.kdma_association[target_kdma['kdma']]
			dist = abs(sampled_target-choice_value)
			if dist < min_dist:
				min_dist = dist 
				selected_action = action

	return selected_action