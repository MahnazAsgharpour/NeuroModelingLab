
import os
import sys
import pandas as pd
from tqdm import tqdm

import numpy as np
from numpy import rad2deg, cross, arccos
from numpy.linalg import norm

from solid2 import cylinder

progname = os.path.basename(sys.argv[0])


class Branch (object):
    def __init__(self, ID, coords, diams, parent=None, children=None):
        self.ID = ID
        self.coords = coords
        self.diams = np.array(diams)
        self.diam = np.mean(diams)
        self.parent = parent
        self.children = children if children is not None else []


def _build_tree(df, start, coords=['x','y','z']):
    idx = start
    path,diams = [],[]
    while True:
        ID = df.loc[idx].name
        path.append(df.loc[idx,coords].to_numpy())
        diams.append(df.loc[idx,'diam'])
        children, = np.where(df.loc[:,'parent'] == ID)
        N_children = len(children)
        if N_children == 1:
            idx = df.index[children[0]]
            continue
        else:
            branch = Branch(df.loc[start].name, np.array(path), diams)
            children_branches = []
            for child in df.index[children]:
                children_branches += _build_tree(df, child)
            for child_branch in children_branches:
                child_branch.parent = branch
                branch.children.append(child_branch)
            break
    return [branch]


def _gather_branches(branch, branch_list):
    if branch is not None:
        branch_list.append(branch)
        for child in branch.children:
            _gather_branches(child, branch_list)


def _make_paths_to_leaf(root, with_diams=True):
    stack_coords_diams = lambda node: np.hstack((node.coords,
                                                 node.diams[:,np.newaxis]))
    if len(root.children) == 0:
        if with_diams:
            return [stack_coords_diams(root)]
        return [root.coords]
    paths = []
    for child in root.children:
        children_paths = _make_paths_to_leaf(child, with_diams)
        for child_path in children_paths:
            if with_diams:
                paths.append(np.vstack((stack_coords_diams(root), child_path)))
            else:
                paths.append(np.vstack((root.coords, child_path)))
    return paths


def _make_scad_cylinder(pt1, pt2, d1, d2, _fn):
    V = pt2 - pt1
    H = norm(V)
    if H == 0:
        raise Exception('pt1 and pt2 must be distinct points')
    # vector to which cylinders created by OpenSCAD are aligned to
    U = np.array([0.,0.,1.])
    # arccos((U@V) / (norm(U) * norm(V))))
    theta = rad2deg(arccos(V[2]/norm(V)))
    assert not np.isnan(theta)
    C = cylinder(H, d1=d1, d2=d2, _fn=_fn)
    W = cross(U,V)
    if norm(W) != 0:
        C = C.rotate(theta, W)
    else:
        if theta == 0:
            pass
        elif theta == 180:
            C = C.rotate(theta, [1,0,0])
        else:
            raise Exception('theta should be either 0 or 180 if the cylinder is parallel to the Z-axis')
    return C.translate(pt1)


def _make_scad_path(path, _fn=5):
    N_pts = len(path)
    scad_path = None
    for i in range(N_pts-1):
        pt1,d1 = path[i,:3],path[i,3]
        pt2,d2 = path[i+1,:3],path[i+1,3]
        try:
            C = _make_scad_cylinder(pt1, pt2, d1, d2, _fn)
            if scad_path is None:
                scad_path = C
            else:
                scad_path += C
        except:
            pass
    return scad_path


def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [-o | --outfile <filename>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-f | --force] [-F | --fix-morpho] [--save-excel]')
    print(prefix + '[--fn | --num-faces <num>] [--no-plot] SWC_file')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':

    i = 1
    N_args = len(sys.argv)
    outdir = None
    force = False
    fn = 5
    fix_morpho = False
    save_excel = False
    with_plot = True

    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
        elif arg in ('-o','--outdir'):
            i += 1
            outdir = sys.argv[i]
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-F', '--fix-morpho'):
            fix_morpho = True
        elif arg in ('--fn', '--num-faces'):
            i += 1
            fn = int(sys.argv[i])
        elif arg == '--save-excel':
            save_excel = True
        elif arg == 'no-plot':
            with_plot = False
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == N_args:
        print(f'{progname}: you must specify an input file')
        sys.exit(1)
    if i == N_args-1:
        swc_file = sys.argv[i]
    else:
        print(f'{progname}: arguments after project name are not allowed')
        sys.exit(1)

    if not os.path.isfile(swc_file):
        print(f'{progname}: {swc_file}: no such file.')
        sys.exit(1)

    if outdir is None:
        outdir = os.path.splitext(os.path.basename(swc_file))[0]

    if os.path.isdir(outdir):
        if not force:
            print(f'{progname}: {outdir} exists: use -f to overwrite.')
            sys.exit(1)
        import glob
        for f in glob.glob(os.path.join(outdir, '*.scad')):
            os.remove(f)
    else:
        os.mkdir(outdir)

    morpho = pd.read_table(swc_file, sep=' ', header=None, index_col=0,
                           names=['type','x','y','z','diam','parent'])

    if fix_morpho:
        coords = ['x','y','z']
        to_drop = []
        for idx in morpho.index:
            if morpho.loc[idx,'parent'] == -1:
                continue
            current = morpho.loc[idx,:]
            parent = morpho.loc[current.parent,:]
            if np.linalg.norm(current[coords] - parent[coords]) < 1e-6:
                # find the points that have the current point as parent
                jdx = morpho.loc[:,'parent'] == current.name
                # set as their parent the parent of the current point
                morpho.loc[jdx,'parent'] = parent.name
                to_drop.append(current.name)
        morpho.drop(to_drop, inplace=True)
        print('Removed duplicate points in morphology.')

    coords = ['x','y','z']
    cylinders = None
    sys.stdout.write('Building full SCAD morphology... ')
    sys.stdout.flush()
    for idx in morpho.index:
        if morpho.loc[idx,'parent'] == -1:
            continue
        current = morpho.loc[idx,:]
        parent = morpho.loc[current.parent,:]
        if norm(current[coords] - parent[coords]) > 1e-3:
            pt1,pt2 = parent[coords].to_numpy(), current[coords].to_numpy()
            d1,d2 = parent.diam, current.diam
            C = _make_scad_cylinder(pt1, pt2, d1, d2, _fn=fn)
            if cylinders is None:
                cylinders = C
            else:
                cylinders += C
    outfile = os.path.join(outdir, f'morphology.scad')
    cylinders.save_as_scad(outfile)
    sys.stdout.write('done.\n')

    leaves = [idx for idx in morpho.index if all(morpho.parent != idx)]
    coords = ['x','y','z']
    tree = _build_tree(morpho, morpho.index[0], coords)
    root = tree[0]
    branches = []
    _gather_branches(root, branches)
    paths = _make_paths_to_leaf(root, with_diams=True)
    N_paths = len(paths)
    assert len(leaves) == N_paths
    print('The morphology contains {} unbranched sections and {} paths from root to leaf.'.\
          format(len(branches), N_paths))

    for i in tqdm(range(N_paths), ascii=True, ncols=80):
        outfile = os.path.join(outdir, f'path_{i+1:03d}.scad')
        _make_scad_path(paths[i], _fn=fn).save_as_scad(outfile)

    if save_excel:
        sys.stdout.write('Saving Excel spreadsheets... ')
        sys.stdout.flush()
        outfile = os.path.join(outdir, 'morphology')
        with pd.ExcelWriter(outfile + '_branches.xlsx') as fid:
            for i,branch in enumerate(branches):
                pd.DataFrame(data=branch.coords).to_excel(fid, sheet_name=f'branch_{i+1}',
                                                          header=False, index=False)
        with pd.ExcelWriter(outfile + '_paths.xlsx') as fid:
            for i,path in enumerate(paths):
                pd.DataFrame(data=path).to_excel(fid, sheet_name=f'path_{i+1}',
                                                 header=False, index=False)
        sys.stdout.write('done.\n')

    if with_plot:
        import matplotlib.pyplot as plt
        scalebar_len = 50
        i,j = 0,2
        limits = np.array([[np.min(morpho.loc[:,coords[i]]), np.max(morpho.loc[:,coords[i]])],
                           [np.min(morpho.loc[:,coords[j]]), np.max(morpho.loc[:,coords[j]])]])
        dx,dy = np.diff(limits, axis=1)
        xscale = limits[0,0] + np.array([0,scalebar_len])
        yscale = limits[1,1] + dy/20 + np.zeros(2)
        ratio = dx/dy
        width = 3
        height = width/ratio[0]
        fig,ax = plt.subplots(1, 2, figsize=(width*2,height), sharex=True, sharey=True)
        cmap = plt.get_cmap('Accent')
        for k,branch in enumerate(branches):
            path = branch.coords
            lw = branch.diam
            ax[0].plot(path[:,i], path[:,j], color=[.3,.3,.3], lw=lw)
            ax[1].plot(path[:,i], path[:,j], color=cmap(k%8), lw=lw)
        ax[0].plot(morpho.loc[leaves,'x'], morpho.loc[leaves,'z'], '.', color='tab:red', ms=1)
        ax[1].plot(morpho.loc[leaves,'x'], morpho.loc[leaves,'z'], '.', color='k', ms=1)
        for k,path in enumerate(paths):
            ax[0].text(path[-1,i], path[-1,j], k+1, fontsize=3)
        for a in ax:
            a.plot(0, 0, 's', color='tab:green', ms=7)
            a.set_aspect('equal', adjustable='box')
            a.axis('off')
        ax[0].plot(xscale, yscale, 'k', lw=2)
        ax[0].text(np.sum(xscale)/2, limits[1,1] + dy/22, r'${}\,\mu m$'.format(scalebar_len),
                   fontsize=10, va='top', ha='center')
        fig.tight_layout()
        outfile = os.path.join(outdir, 'morphology.pdf')
        plt.savefig(outfile)


