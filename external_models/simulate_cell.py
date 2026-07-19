
import os
import sys
import json
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937

progname = os.path.basename(sys.argv[0])


def OU(dt, mean, stddev, tau, N, random_state=None):
    if random_state is not None:
        rnd = random_state.normal(size=N)
    else:
        rnd = np.random.normal(size=N)
    const = 2 * stddev**2 / tau
    mu = np.exp(-dt / tau)
    coeff = np.sqrt(const * tau / 2 * (1 - mu ** 2))
    ou = np.zeros(N)
    ou[0] = mean
    for i in range(N-1):
        ou[i+1] = mean + mu * (ou[i] - mean) + coeff * rnd[i]
    return ou


def usage(exit_code=None):
    prefix = 'usage: {}'.format(progname)
    print(f'{prefix} [-o | --output <out_file>] [-f | --force] [--no-plots] [--save-voltage-traces] config_file')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':

    i = 1
    n_args = len(sys.argv)
    outfile = 'simulation.npz'
    force = False
    with_plots = True
    save_voltage_traces = False
    
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
        elif arg in ('-o', '--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--no-plots':
            with_plots = False
        elif arg == '--save-voltage-traces':
            save_voltage_traces = True
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == n_args:
        print(f'{progname}: you must specify a configuration file.')
        sys.exit(1)

    config_file = sys.argv[i]
    if i != n_args-1:
        print(f'{progname}: additional arguments after configuration file not allowed.')
        sys.exit(1)

    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)

    if os.path.isfile(outfile) and not force:
        print('{}: {}: file exists, use -f to overwrite'.format(progname, outfile))
        sys.exit(1)

    from neuron import h

    config = json.load(open(config_file,'r'))
    cell_type = config['cell_type']
    h.celsius = config['sim']['celsius'] if 'celsius' in config['sim'] else 34
    
    import dbbs_models
    cell = getattr(dbbs_models, f'build_{cell_type}_cell')()

    tstop = config['sim']['dur']
    dt = config['sim']['dt'] if 'dt' in config['sim'] else h.dt

    for stim_type,stim_pars in config['stim'].items():
        if stim_pars['dur'] == 0:
            continue

        # we might need these later
        time = np.r_[0 : tstop : dt]
        t_vec = h.Vector(time)

        if stim_type == 'Efield':
            from utils import compute_Ve_over_cell

            ### compute the extracellular potential for each segment
            spherical_to_cartesian = lambda r,th,ph: np.array([r*np.sin(th)*np.cos(ph),
                                                               r*np.sin(th)*np.sin(ph),
                                                               r*np.cos(th)])
            Efield = lambda X: spherical_to_cartesian(stim_pars['mag'],
                                                      stim_pars['theta'],
                                                      stim_pars['phi'])
            Ve,cell_secs,points = compute_Ve_over_cell(cell, Efield, full_output=True)
            assert all([ve.size==sec.nseg for ve,sec in zip(Ve,cell_secs)])
            N_secs = len(cell_secs)

            ### plot the morphology with the values of extracellular potentials
            if with_plots:
                import matplotlib.pyplot as plt
                from matplotlib.colors import CenteredNorm, TwoSlopeNorm
                from dlutils.morpho import Tree
                from dlutils.graphics import plot_tree
                data = np.vstack(list(points.values()))
                PTS,V = data[:,:3],data[:,-1]*1e3
                norm = TwoSlopeNorm(vmin=V.min(), vcenter=0, vmax=V.max())
                tree = Tree(config['morphology'])
                height = 10
                width = height * max(0.2, tree.xz_ratio)
                fig,ax = plt.subplots(1, 1, figsize=(width,height))
                plot_tree(tree, coords='xz', cmap='coolwarm', norm=norm, points=PTS, values=V,
                          show_cbar=True, cbar_ticks=5, cbar_label=r'$V_{\mathrm{extra}}$ (mV)', ax=ax, diam_coeff=10)
                ax.axis('off')
                fig.tight_layout()
                plt.savefig(os.path.splitext(outfile)[0] + '_Vextra_morpho.pdf')

            stim = np.zeros_like(time)
            idx = (time >= stim_pars['delay']) & (time < stim_pars['delay']+stim_pars['dur'])
            stim[idx] = 1
            E_vecs = [[] for _ in range(N_secs)]
            for i,sec in enumerate(cell_secs):
                sec.insert('extracellular')
                for j,seg in enumerate(sec):
                    vec = h.Vector(stim * Ve[i][j] * 1e3)
                    vec.play(seg._ref_e_extracellular, t_vec, 1)
                    E_vecs[i].append(vec)
        
        elif stim_type == 'IClamp':
            Iclamp = h.IClamp(cell.soma[0](0.5).__neuron__())
            if 'amp' in stim_pars:
                Iclamp.delay = stim_pars['delay']
                Iclamp.dur = stim_pars['dur']
                Iclamp.amp = stim_pars['amp']
            elif 'mean' in stim_pars:
                Iclamp.dur = tstop
                mu,sigma,tau = stim_pars['mean'],stim_pars['stddev'],stim_pars['tau']
                rs = RandomState(MT19937(SeedSequence(stim_pars['seed']))) if \
                    'seed' in stim_pars else None
                ou = OU(dt, mu, sigma, tau, time.size, rs)
                idx = (time < stim_pars['delay']) | (time > stim_pars['delay']+stim_pars['dur'])
                ou[idx] = 0
                OU_vec = h.Vector(ou)
                OU_vec.play(Iclamp._ref_amp, t_vec, 1)
        else:
            raise Exception(f"Unknown stimulation type '{stim_type}'")

    rec = {'time':        h.Vector(),
           'Vsoma':       h.Vector(),
           'spike_times': h.Vector()}
    rec['time'].record(h._ref_t)
    rec['Vsoma'].record(cell.soma[0](0.5)._ref_v)
    apc = h.APCount(cell.soma[0](0.5).__neuron__())
    apc.thresh = 0
    apc.record(rec['spike_times'])

    h.dt = dt
    h.tstop = tstop
    h.cvode_active(0)
    h.run()

    t,Vsoma = np.array(rec['time']), np.array(rec['Vsoma'])

    kwargs = {'config': config,
              'spike_times': np.array(rec['spike_times']) if len(rec['spike_times']) > 0 else np.array([])}
    if save_voltage_traces:
        kwargs.update({'time': t, 'Vsoma': Vsoma})
    np.savez_compressed(outfile, **kwargs)

    if with_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig,ax = plt.subplots(1, 1, figsize=(5,3.5))
        ax.plot(t, Vsoma, 'k', lw=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Vm (mV)')
        ax.grid(which='major', axis='y', lw=0.5, ls=':', color=[.6,.6,.6])
        sns.despine()
        fig.tight_layout()
        plt.savefig(os.path.splitext(outfile)[0] + '_Vsoma.pdf')

