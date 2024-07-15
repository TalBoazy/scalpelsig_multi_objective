import numpy as np



def kl_divergence(target, out, smooth=1e-10):
    n = out.shape[0]
    # smoothing before calculating, should check if there is a better solution.
    target += smooth
    out += smooth
    return np.sum(out * np.log(out / target)) / n


def calculate_windows_exposures(batch, e):
    n = batch.shape[0]
    w = batch.shape[1]
    k = e.shape[0]
    pi = np.zeros(shape=(n,w,k))
    log_likelihood = np.sum(batch[..., np.newaxis] * e[np.newaxis, np.newaxis, :, :], axis=3)
    selected_sig = np.argmax(log_likelihood, axis=2)
    n_idx, w_idx = np.indices(selected_sig.shape)
    pi[n_idx, w_idx, selected_sig] = 1
    return np.exp(pi)


def select_top_k_features(window_manager, k, e, real_exposures):
    selected_pi = np.zeros(shape=real_exposures.shape)
    e = np.log(e)
    # a list of 3d tuples of the structure (chrom,batch,index)
    selected = []
    best_pi = None
    best_score = np.inf
    best_name = None
    for i in range(k):
        print("iteration: {}".format(k))
        prev_chrom = -1
        for key in window_manager.window_index:
            (chrom, batch_index) = eval(key)
            if chrom != prev_chrom:
                print("currently processing chrom " + chrom)
                prev_chrom = chrom
            batch = window_manager.get_batch(chrom, batch_index)[window_manager.samples].astype('int')
            pi = calculate_windows_exposures(batch, e)
            batch_selected = [s[2] for s in selected if (s[0], s[1]) == (chrom, batch_index)]
            tot_pi = selected_pi[:,:,np.newaxis]+pi
            tot_pi = tot_pi/tot_pi.sum(axis=2)
            if len(batch_selected):
                mask = np.ones(batch.shape[1], dtype=bool)
                mask[batch_selected] = False
                tot_pi[:,mask,:]= 10
            scores = kl_divergence(real_exposures,tot_pi)
            best_window = np.argmin(scores)
            if scores[best_window]<best_score:
                best_pi = pi[:,best_window,:]
                best_score = scores
                best_name = (chrom, batch_index, best_window)
    selected.append(best_name)
    selected_pi += best_pi


# a = np.random.randint(0, 10, size=(10, 5, 20)).astype(np.float64)
#
# # Generate a random (10, 20) matrix with integer values between 0 and 10
# b = np.random.randint(0, 10, size=(10, 20)).astype(np.float64)
#
# kl_divergence(b,a)