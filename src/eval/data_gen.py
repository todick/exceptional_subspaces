import numpy as np
from itertools import combinations, islice
from sklearn.neighbors import KernelDensity

'''This function generates a subgroup distribution based on the specified distribution type.'''
def get_subgroup_dist(condition,dist):
    if dist == "normal":
        return np.random.normal(loc=1.5, scale=0.5,size=(np.sum(condition)))
    if dist == "bi_modal":
        mask = np.random.choice(2,size=(np.sum(condition)))
        mode1 = np.random.normal(loc=-1.5, scale=0.5,size=(np.sum(condition)))*mask
        mode2 = np.random.normal(loc=1.5, scale=0.5,size=(np.sum(condition)))*(1-mask)
        return mode1+mode2
    if dist == "beta":
        return np.random.beta(a=0.2,b=0.2,size=(np.sum(condition)))*1.2
    if dist == "rayleigh":
        return np.random.rayleigh(scale=2, size=(np.sum(condition)))
    if dist == "uniform":
        return np.random.uniform(low=0.5, high=1.5, size=(np.sum(condition)))
    if dist == "exponential":
        return np.random.exponential(scale=0.5,size=(np.sum(condition)))
    if dist=="cauchy":
        return np.random.standard_cauchy(size=(np.sum(condition)))

'''This function generates a distribution for the non-subgroup samples. The distribution is
based on the specified distribution type and is generated using the same parameters as the
subgroup distribution.'''
def get_distribution(n_samples, base_dist, seed = None):
    assert base_dist in ["rayleigh","cauchy","normal", "bi_modal", "beta", "uniform", "exponential"]
    if seed:
        np.random.seed(seed)
    Y = get_subgroup_dist(np.ones(n_samples).astype(bool), base_dist)
    return Y                   

'''This function adds features to the data which also include constraints with random features.
This can rapidly increase the number of samples.'''
def add_features(data, target, n_features, n_constraints = 3, seeded_sgs = [], seed = None):
    if seed:
        np.random.seed(seed)
    data = data.copy()
    target = target.copy()
    for _ in range(n_features):
        # sample three random features to be constrained with
        constraint = np.zeros(data.shape[1]).astype(bool)
        constraint[np.random.choice(data.shape[1], n_constraints, replace=False)] = 1
        # add the feature as a new column. If the constraint is not met, the feature is set to 0. 
        # If the constraint is met, the feature is set to 0, the sample is cloned with the same 
        # target value and the feature is set to 1 in the clone.
        target = np.concatenate([target, target[data[:,constraint].prod(axis=1)!=0]])
        seeded_sgs = [np.concatenate([sg, sg[data[:,constraint].prod(axis=1)!=0]]) for sg in seeded_sgs]
        data_feature_pos = data[data[:,constraint].prod(axis=1)!=0]
        data_feature_pos = np.hstack([data_feature_pos, np.ones((data_feature_pos.shape[0],1))])
        data = np.hstack([data, np.zeros((data.shape[0],1))])
        data = np.concatenate([data, data_feature_pos])
    return data, target, seeded_sgs

'''This function adds features to the data without any constraints. The features are set to 1 with a probability of 0.5.'''
def add_features_unconstrained(data, n_features, seed = None):
    if seed:
        np.random.seed(seed)
    data = data.copy()
    for _ in range(n_features):
        # add the feature as a new column. The feature is set to 1 with a probability of 0.5
        data = np.hstack([data, np.random.choice([0,1],size=(data.shape[0],1))])
    return data

'''This function increases the number of samples by duplicating existing samples and adding random noise (only works for integer factors).'''
def add_samples(data, target, factor, noise = 0.05, seed = None):
    if seed:
        np.random.seed(seed)
    data_new = data.copy()
    target_new = target.copy()
    for _ in range(factor-1):
        data_new = np.concatenate([data_new, data])
        target_new = np.concatenate([target_new, target + np.random.normal(0, noise, size=(target.shape[0],))])
    return data_new, target_new

'''This function seeds a subgroup in the data without changing the data distribution.'''
def seed_subgroup(data, target, n_conditions, target_dist, min_rule_size = 0.1, max_rule_size = 0.2,
                   n_tries = 100000, n_groups = 2, rel_group_width = 0.05, max_overlap = 0.1,
                   feature_names = None, seed = None):
    assert target_dist in ["rayleigh","cauchy","normal", "bi_modal", "beta", "uniform", "exponential"]
    assert n_conditions <= data.shape[1]
    
    Y = target.copy()
    if seed:
        np.random.seed(seed)
    if feature_names is None:
        feature_names = [f"X_{i}" for i in range(data.shape[1])]
    
    candidates = {}
    for f in range(data.shape[1]):
        values = [v for v in np.unique(data[:,f]) if (data[:,f] == v).sum() > min_rule_size*data.shape[0]]
        if len(values) > 1:
            if set(np.unique(data[:,f])) == {0,1}:
                values = [1]
            candidates[f] = list(values)
    if len(candidates) < n_conditions:
        print("Could not find enough features to seed the subgroup.")
        return None, None, None
    
    subgroups = []
    rules = []
    in_subgroup = np.zeros(data.shape[0])
    min_size = min_rule_size * data.shape[0]
    max_size = max_rule_size * data.shape[0]

    # precompute non-overlapping intervals for all subgroups
    Y_min, Y_max = np.min(Y), np.max(Y)
    interval = (Y_max - Y_min) / n_groups
    subgroup_intervals = [(Y_min + i * interval, Y_min + (i + 1) * interval) for i in range(n_groups)]

    for i in range(n_groups):
        max_allowed_overlap = max_overlap * in_subgroup.sum()
        # Generate all possible combinations of feature-value pairs
        feature_combinations = list(islice(combinations(candidates.keys(), n_conditions), n_tries))
        # Generate interactions by assigning each feature in the feature_combinations a random value
        interactions = []
        for combination in feature_combinations:
            # If all values for a certain feature are present in the same number of samples, select a random value.
            # Otherwise, select all possible values. This will lead to multiple interactions with the same features.
            interactions_comb = [[]]
            for f in combination:
                len_v1 = (data[:,f] == candidates[f][0]).sum()
                if all([(data[:,f] == v).sum() == len_v1 for v in candidates[f]]):
                    values = [np.random.choice(candidates[f])]
                else:
                    values = candidates[f]
                interactions_comb = [interaction + [(f,v)] for interaction in interactions_comb for v in values]
            interactions.extend(interactions_comb)
        np.random.shuffle(interactions)
        print(f"Trying to seed subgroup {i+1}/{n_groups}, testing up to {len(interactions)} interactions.") 
        for interaction in interactions:        
            subgroup = np.ones(data.shape[0])   
            for f, value in interaction:
                subgroup = np.logical_and(subgroup, data[:,f] == value)

            # Check if the subgroup meets the size and overlap constraints
            subgroup_size = np.sum(subgroup)
            if subgroup_size < min_size or subgroup_size > max_size:
                continue
            overlap = np.logical_and(subgroup, in_subgroup).sum()
            if overlap > max_allowed_overlap:
                continue

            # Make a collinearity check for all features in the interaction
            X = data[:,[f for f, _ in interaction]]
            if np.linalg.matrix_rank(X) < n_conditions:
                continue
            
            in_subgroup = np.logical_or(in_subgroup, subgroup)
            Y_subgroup = get_subgroup_dist(subgroup, target_dist)
            # scale subgroup to unit size
            Y_subgroup = Y_subgroup - np.mean(Y_subgroup)
            Y_subgroup = Y_subgroup / np.max(np.abs(Y_subgroup))
            # scale subgroup to fit in with the data
            Y_scale = np.max(Y) - np.min(Y)
            Y_subgroup = Y_subgroup * Y_scale * rel_group_width / 2
            # shift the subgroup to fit in the interval
            interval_start, interval_end = subgroup_intervals[i]
            y_min, y_max = np.min(Y_subgroup), np.max(Y_subgroup)
            # The allowed shift range so that the whole subgroup fits in the interval
            shift_min = interval_start - y_min
            shift_max = interval_end - y_max
            if shift_max <= shift_min:
                # Not enough space, skip this interaction
                continue
            shift = np.random.uniform(shift_min, shift_max)
            Y_subgroup = Y_subgroup + shift
            Y[subgroup] = Y_subgroup
            subgroups.append(subgroup)
            rules.append(" AND ".join([f"{feature_names[f]} = {v}" for f,v in interaction]))

            # remove the selected values from the candidate list
            for f, v in interaction:
                candidates[f].remove(v)
                if len(candidates[f]) == 0:
                    del candidates[f]                    
            break
        
        if len(subgroups) < i+1:
            print(f"Could not find a suitable rule to seed subgroup {i+1}.")
            return None, None, None
        
    return Y, subgroups, rules                         

'''This function generates a sampleset fitting the given distribution.'''
def scale_and_seed_subgroup(n_samples, n_features, original_target, numeric = 0.1, numeric_values = [1,2,4,8,16], sg_constraints = 4, sg_dist = "normal", sg_size = 0.1, seed = None):
    if seed:
        np.random.seed(seed)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(original_target)
    Y = kde.sample(n_samples)
    Y = Y.flatten()
    X_numeric = np.random.choice(numeric_values, size=(n_samples, int(n_features*numeric)), replace=True)
    X_binary = np.random.choice([0,1], size=(n_samples, int(n_features*(1-numeric))), replace=True)
    # make sure that features add up to the correct number of features
    while X_numeric.shape[1] + X_binary.shape[1] < n_features:
        X_binary = np.hstack([X_binary, np.random.choice([0,1], size=(n_samples, 1), replace=True)])
    X = np.hstack([X_numeric, X_binary])

    subgroup_features = np.random.choice(n_features, sg_constraints, replace=False)
    subgroup_values = []
    for f in subgroup_features:
        values = np.unique(X[:,f])
        if len(values) > 1:
            subgroup_values.append(np.random.choice(values))
        else:
            subgroup_values.append(values[0])
    subgroup = np.zeros(n_samples).astype(bool)
    subgroup[np.random.choice(n_samples, int(n_samples*sg_size), replace=False)] = 1
    for i in range(len(subgroup_features)):
        X[subgroup, subgroup_features[i]] = subgroup_values[i]
    
    subgroup = np.ones(n_samples).astype(bool)
    for i in range(len(subgroup_features)):
        subgroup = np.logical_and(subgroup, X[:,subgroup_features[i]] == subgroup_values[i])

    Y_subgroup = get_subgroup_dist(subgroup, sg_dist)
    Y_subgroup = Y_subgroup - np.mean(Y_subgroup)
    Y_subgroup = Y_subgroup / np.max(np.abs(Y_subgroup))
    Y_subgroup = Y_subgroup * np.max(Y) * 0.05
    shift = np.random.uniform(np.min(Y), np.max(Y))
    Y_subgroup = Y_subgroup + shift
    Y[subgroup] = Y_subgroup

    subgroup_rule = " AND ".join([f"X_{subgroup_features[i]}=={subgroup_values[i]}" for i in range(len(subgroup_features))])
    return X, Y, [subgroup], [subgroup_rule]