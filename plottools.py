import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_metrics(d, namefile="metrics.png"):
    """
    Plot a dictionary of metrics
    """
    fig, axes = plt.subplots(nrows=len(d), ncols=1)
    ax_list = axes.flatten()

    for metric, ax in zip(d, ax_list):
        ax.set_title(metric)
        ax.plot(d[metric])

    fig.tight_layout()
    plt.savefig(namefile)
    plt.clf()
    plt.close()


def plot_batch(batch, namefile=None):
    fig, axes = plt.subplots(4, 8,
                             subplot_kw={'xticks': [], 'yticks': []})
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in zip(range(32), axes.flat):
        ax.imshow(((batch[i]+1)*(255./2)).astype(np.uint8),
                  interpolation=None,
                  cmap="gray")
    fig.tight_layout()
    if namefile is None:
        plt.show()
    else:
        plt.savefig(namefile)
    plt.clf()
    plt.close()

def plot_all(loss_dict, acc_dict, batch_true, batch_gen, 
             namefile,score_true=None, score_gen=None,
             data_per_step=None):
    gs = gridspec.GridSpec(4, 4)
    #gs.update(left=0.05, right=0.48, wspace=0.05)

    ax_loss = plt.subplot(gs[0, :])
    ax_acc = plt.subplot(gs[1, :])
    ax_true = plt.subplot(gs[2:,:2])
    ax_gen = plt.subplot(gs[2:,2:])

    for _, loss in loss_dict.items():
        n = len(loss)
    x = np.arange(n)
    if data_per_step is not None:
        x = data_per_step * x

    # Loss plot
    ax_loss.set_title("Loss ")
    x_loss = np.arange(500)
    ax_loss.set_xlim([0., x.max()])
    for label, loss in loss_dict.items():
        ax_loss.plot(x, loss, label=label, linewidth=.6, alpha=0.7)
    ax_loss.set_yscale('log')
    #ax_loss.set_xlabel('Mini-batch-number')
    #ax_loss.set_ylabel('Loss (log-scale)')
    ax_loss.legend(loc="upper right",fontsize=4.)

    # Accuracy plot
    ax_acc.set_title("Accuracy")
    ax_acc.grid(axis='y', color='k', linewidth=0.2)
    
    for label, acc in acc_dict.items():
        ax_acc.plot(x, acc, label=label, linewidth=.6, alpha=0.7)
    #ax_acc.plot(x, half, 'k--', linewidth=.5)
    ax_acc.set_xlim([0., x.max()])
    ax_acc.set_ylim([-0.01, 1.01])
    #ax_acc.set_xlabel('Mini-batch-number')
    #ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc="upper right",fontsize=4.)

    # Images plot
    cmap = plt.get_cmap("brg")
    
    # True images plot
    if score_true is not None:
        
        batch_true = batch_true[np.argsort(score_true),:,:,:]
        score_true = np.sort(score_true)
        batch_color = np.zeros(batch_true.shape[:-1]+(4,))
        batch_color[:,:,:,-1] = (batch_true[:,:,:,0] + 1)/2
        for k in range(score_true.shape[0]):
            batch_color[k,:,:,:-1] = cmap(score_true[k])[:3]
        batch_true = batch_color
        
    
    
    ax_true.set_title("Pictures from dataset")
    ax_true.axis('off')
    space_pxl = 2
    #if batch_true.ndim == 4:
    #    n, u, v, c = batch_true.shape
    #    s = int(np.sqrt(n))
    #    X = np.zeros((s*u + (s+1)*space_pxl, s*v + (s+1)*space_pxl, c))
    
    n, u, v, c = batch_true.shape
    s = int(np.sqrt(n))
    X = np.zeros((s*u + (s+1)*space_pxl, s*v + (s+1)*space_pxl, c))


    for k in range(s):
        for l in range(s):
            X[k*(u+space_pxl)+space_pxl:(k+1)*(u+space_pxl),
              l*(v+space_pxl)+space_pxl:(l+1)*(v+space_pxl),:] = \
              batch_true[k*s+l]

    #X = ((X + 255/2)*255/2).astype(np.uint8)
    if c==1:
        ax_true.imshow(X, aspect='equal', interpolation='none',cmap="gray")
    else:
        ax_true.imshow(X, aspect='equal', interpolation='none')


    # Generated images plot
    if score_gen is not None:
        batch_gen = batch_gen[np.argsort(score_gen)]
        score_gen = np.sort(score_gen)
        batch_color = np.zeros(batch_gen.shape[:-1]+(4,))
        batch_color[:,:,:,-1] = (batch_gen[:,:,:,-1] + 1)/2
        for k in range(score_gen.shape[0]):
            batch_color[k,:,:,:-1] = cmap(score_gen[k])[:3]
        batch_gen = batch_color
        
    ax_gen.set_title("Generated pictures")
    ax_gen.axis('off')
    space_pxl = 2
    #if batch_true.ndim == 4:
    #    n, u, v, c = batch_true.shape
    #    s = int(np.sqrt(n))
    #    X = np.zeros((s*u + (s+1)*space_pxl, s*v + (s+1)*space_pxl, c))
    #else:
    n, u, v, c = batch_true.shape
    s = int(np.sqrt(n))
    X = np.zeros((s*u + (s+1)*space_pxl, s*v + (s+1)*space_pxl, c))


    for k in range(s):
        for l in range(s):
            X[k*(u+space_pxl)+space_pxl:(k+1)*(u+space_pxl),
              l*(v+space_pxl)+space_pxl:(l+1)*(v+space_pxl)] = \
              batch_gen[k*s+l]

    #X = ((X + 255/2)*255/2).astype(np.uint8)
    if c==1:
        ax_gen.imshow(X, aspect='equal', interpolation='none',cmap="gray")
    else:
        ax_gen.imshow(X, aspect='equal', interpolation='none')
    #gs.tight_layout()
    gs.update(hspace=1.)
    plt.savefig(namefile, format='pdf')
    
    
def plot_latent_space(x, y, encoder, batch_size, namefile):
    x_encoded = encoder.predict(x, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_encoded[:, 0], x_encoded[:, 1], c=y)
    plt.colorbar()
    plt.savefig(namefile)
    
def plot_manifold(generator, batch_size, ndigits=15, digit_size=28):
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n)) 
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig("img/vae_manifold.png")
