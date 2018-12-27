import time
import numpy as np
import torch

# load data

dataloaders = {x: torch.utils.data.DataLoader(
	datasets[x])}


# build graph

# run training

def get_logs(key, _run):
	logs = _run.info.get('logs', {})

def train(log_dir, max_epoch):
	best_valid_loss = np.inf
	best_valid_epoch = 0

	for epoch in range(1, max_epoch+1):
		t = time.time()

		# log and print output
		log_dict = run_epoch()
		log_log_dict('training', log_dict)
		print("\n" + 80 * "%" + "    EPOCH {}   ".format(epoch) + 80 * "%")
        print_log_dict(log_dict, 'Train', t, dt, s_loss_weights, dt_s_loss_weights)

        if log_dict['loss'] < best_valid_loss:
        	best_valid_loss = log_dict['loss']
        	best_valid_epoch = epoch

            # save model
            save_destination = torch.save(model.state_dict(), os.path.abspath(os.path.join(log_dir, 'best')))
            print("    Saved to:", save_destination)

        if epoch in save_epochs:
            save_destination = torch.save(model.state_dict(), os.path.abspath(os.path.join(log_dir, 'epoch_{}'.format(epoch))))
            print("    Saved to:", save_destination)

        best_valid_loss = min(best_valid_loss, log_dict['loss'])

        if best_valid_loss < np.min(get_logs('validation.loss')[-training['max_patience']:]):
            print('Early Stopping because validation loss did not improve for {} epochs'.format(training['max_patience']))
            break

        if np.isnan(log_dict['loss']):
            print('Early Stopping because validation loss is nan')
            break