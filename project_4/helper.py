import numpy as np

def transform_env(env, reward, policy):
    for state in range(env.nS):
        for action in range(env.nA):
            for transition_policy, s_prime, transition_reward, _ in env.P[state][action]:
                reward[state,action], policy[action,state,s_prime] =  transition_reward, policy[action,state,s_prime] + transition_policy
    
    return reward, policy/np.sum(policy[action,state,:])