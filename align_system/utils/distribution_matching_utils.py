from align_system.utils import kde_utils
import numpy as np

def match_to_target_kde_sample(alignment_target, available_actions, kde_norm, num_kde_samples=1):
	'''
	Returns the choice with min average distance to random samples from the target KDE
	'''
	target_kdma = alignment_target.kdma_values[0] # TODO extend this to multi-KDMAs
	target_kde = kde_utils.load_kde(target_kdma, kde_norm)
	sampled_targets = target_kde.sample(num_kde_samples)
	
	selected_action = None
	min_dist = float('inf')
	for action in available_actions:
		if action.kdma_association is not None:
			sum_of_dists = 0
			choice_value = action.kdma_association[target_kdma['kdma']]
			for sampled_target in sampled_targets:
				sum_of_dists += abs(sampled_target-choice_value)
			dist = sum_of_dists/num_kde_samples
			if dist < min_dist:
				min_dist = dist 
				selected_action = action

	return selected_action


def max_likelihood_matching(alignment_target, available_actions, kde_norm):
	'''
	MLE - Returns the choice with the max likelihood under the target KDE
	'''
	selected_action = None
	target_kdma = alignment_target.kdma_values[0] # TODO extend this to multi-KDMAs
	target_kde = kde_utils.load_kde(target_kdma, kde_norm)

	max_likelihood = 0
	for action in available_actions:
		if action.kdma_association is not None:
			choice_value = action.kdma_association[target_kdma['kdma']]
			log_likelihood = target_kde.score_samples(np.array([choice_value]).reshape(-1, 1))[0]
			likelihood = np.exp(log_likelihood)
			if likelihood > max_likelihood:
				max_likelihood = log_likelihood
				selected_action = action
	
	return selected_action