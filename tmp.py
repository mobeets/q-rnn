# training params
batch_size = 8
random_update = True # If you want to do random update instead of sequential update
lookup_step = 100 # number of time steps in sampled episode
max_epi_len = 600 # max number of time steps used in sample episode
max_epi_num = 100 # max number of episodes remembered
gamma = 0.9 # reward discount factor

# create model
include_prev_reward = True
include_prev_action = True

def main():
	params = dict((x,y) for x,y in globals().items() if not x.startswith('__') and not callable(y))
	print(params)

if __name__ == '__main__':
	main()
