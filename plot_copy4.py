"""
Author: Nianzu Ethan Zheng
Datetime: 2018-2-2
Place: Shenyang China
Copyright
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import imageio
from glob import glob
import os
from matplotlib import gridspec, patches

plt.style.use('classic')

# from cycler import cycler
# import matplotlib as mpl
# plt.style.use('classic-znz')
# mpl.rcParams['lines.linewidth'] = 1
# mpl.rc('legend', labelspacing=0.05, fontsize='medium')
# mpl.rcParams['legend.labelspacing'] = 0.05
# mpl.rc('axes', prop_cycle=cycler(
#     'color', ['#8EBA42', '#988ED5', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
# ))
#                                                                             # E24A33 : red
#                                                                             # 348ABD : blue
#                                                                             # 988ED5 : purple
#                                                                             # 777777 : gray
#                                                                             # FBC15E : yellow
#                                                                             # 8EBA42 : green
#                                                                             FFB5B8 : pink
#
#
# WHITE	     #FFFFFF	RGB(255, 255, 255)
# SILVER	 #C0C0C0	RGB(192, 192, 192)
# GRAY	     #808080	RGB(128, 128, 128)
# BLACK      #000000	RGB(0, 0, 0)
# RED	     #FF0000	RGB(255, 0, 0)
# MAROON	 #800000	RGB(128, 0, 0)
# YELLOW	 #FFFF00	RGB(255, 255, 0)
# OLIVE	     #808000	RGB(128, 128, 0)
# LIME	     #00FF00	RGB(0, 255, 0)
# GREEN	     #008000	RGB(0, 128, 0)
# AQUA	     #00FFFF	RGB(0, 255, 255)
# TEAL	     #008080	RGB(0, 128, 128)
# BLUE     	 #0000FF	RGB(0, 0, 255)
# NAVY	     #000080	RGB(0, 0, 128)
# FUCHSIA	 #FF00FF	RGB(255, 0, 255))
# PURPLE	 #800080	RGB(128, 0, 128))



# mpl.rc('xtick', labelsize='small')




class Visualizer:
    def __init__(self):
        self.ms = 2

    def tsplot(self, x, name, dir):
        """Time series plot
            Name
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        plt.plot(x, '-r*', ms=self.ms, linewidth=1)
        plt.legend(name, fontsize=10)
        plt.ylabel(name + 'loss per epoch')
        plt.xlabel('Epoch', fontsize=9)
        plt.savefig('{}/{}.png'.format(dir, 'Loss-' + name),  dpi=500)
        plt.close()

    def dyplot(self, x, y, name, dir):
        """double y axis plot
            Name: [filename, ylabel1, ylabel2]
        """
        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=500, facecolor='white')
        ax1.plot(x, '-b*', ms=self.ms, linewidth=1)
        ax1.set_xlabel('Epoch', fontsize=9)
        ax1.set_ylabel(name[1]+'Loss per Epoch', fontsize=9, color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot( y, '-r*', ms=self.ms, linewidth=1)
        ax2.set_ylabel(name[2]+'Loss per Epoch', fontsize=9, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        plt.savefig('{}/{}.png'.format(dir, name[0]))
        plt.close()

    def cplot(self, x, y, name, dir):
        """contrast diagram
            Name:   [filename, xlabel, ylabel, legend]
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        plt.plot(x, '-r*', ms=2.2, linewidth=1.5)
        plt.plot(y, '-b*', ms=2.2, linewidth=1.5)
        plt.legend(name[3], fontsize=10)
        plt.ylabel(name[2] + ' per epoch')
        plt.xlabel(name[1], fontsize=9)
        plt.savefig('{}/{}.png'.format(dir, 'Loss-' + name[0]))
        plt.close()

    def mtsplot(self, y, name, dir):
        """multiple diagrams
        name: format [title, y_label, legend]
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        colors = ["#E24A33", "#348ABD", "#8EBA42", "#ac02ab", "#aba808", "#05aaa8", "#151515", "#94a169", "#bec9cd",
                  "#6a6551"]
        # E24A33 : red
        # 348ABD : blue
        # 988ED5 : purple
        # 777777 : gray
        # FBC15E : yellow
        # 8EBA42 : green
        # FFB5B8 : pink
        for n in range(y.shape[1]):
            plt.plot(y[:, n], c=colors[n], linewidth=1, marker='*', markersize=self.ms)
        plt.legend(name[2:], fontsize=10)
        plt.xlabel('Epochs', fontsize=9)
        plt.ylabel(name[1] + ' value', fontsize=9)
        plt.title(name[0])
        plt.savefig('{}/{}.png'.format(dir, name[0]))
        plt.close()

    def kdeplot(self, x, y, _dir, name='latent_space'):
        sns.set(style='dark')
        f, ax = plt.subplots(figsize=(6, 6))
        cmap = sns.cubehelix_palette(n_colors=6, start=1, light=1, rot=0.4, as_cmap=True)
        sns.kdeplot(x, y, cmap='Blues', shade=True, cut=5, ax=ax)
        f.savefig('{}/{}.png'.format(_dir, name))
        plt.close()

    def jointplot(self, x, y, _dir, kind="scatter", name="latent_space"):
        sns.set(style='white')
        g = sns.jointplot(x, y, kind=kind, size=6, space=0, color='b')
        plt.savefig('{}/{}.png'.format(_dir, name))

    def create_animated_gif(self, pattern_path, name, _dir):
        files = glob(pattern_path)
        imgs = []
        for file in files:
            img = imageio.imread(file)
            imgs.append(img)
            print(file," has been added")
        save_path = os.path.join(_dir, name)
        imageio.mimsave(save_path, imgs)
        print("The gif has been done and put into {}".format(save_path))

    def scatter_z(self, z_batch, color, path=None, filename='z'):
        if path is None:
            raise Exception('please try a valid folder')

        plt.close("all")
        fig = plt.gcf()
        fig.set_size_inches(20.0, 16.0)
        plt.scatter(z_batch[:, 0], z_batch[:, 1], c=color, s=600, marker="o", edgecolors='none')
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.savefig("{}/{}.png".format(path, filename))

    def scale_norm(self, arr):
        arr = arr - arr.min()
        scale = (arr.max() - arr.min())
        return arr / scale

    def img_grid(self, x, t, height, width, path="./pick", name=""):
        """Create a canvas to picture all the example  very quickly
            x         [batch_size, xdim]
        """
        plt.close('all')
        padsize = 1
        padval = .5
        ph = height + 2 * padsize
        pw = width + 2 * padsize
        batch_size = x.shape[0]
        N = int(np.sqrt(batch_size))
        x = x.reshape((N, N, height, width))
        img = np.ones((N * ph, N * pw)) * padval
        for i in range(N):
            for j in range(N):
                startr = i * ph + padsize
                endr = startr + height
                startc = j * pw + padsize
                endc = startc + width
                img[startr:endr, startc:endc] = x[i, j, :, :]

        plt.matshow(img, cmap=plt.cm.gray)
        path = path + "/%s_%s.png" % (name, str(t).zfill(3))
        plt.savefig(path)

    def annotated_img_grid(self, imgs, iter, width, height, region, name="draw"):
        """Plot the images with position where the picture changes
        --------------------------------------------
        iter              i th iteration
        x                 Shape:(batch_size, x_dims)
        width, height     Figure size
        region           (batch_size, 4)
        ------
        """

        plt.close("all")
        batch_size = imgs.shape[0]
        Nimg = int(np.sqrt(batch_size))
        imgs = imgs[:Nimg ** 2, :].reshape([Nimg, Nimg, -1])
        region = region[:Nimg ** 2].reshape([Nimg, Nimg, -1])

        self.fast_animated_grid(imgs, region, height, width,
                                path="./pick", name="%s_%s" % (name, str(iter).zfill(2)), mode="spatial")

    def mesh2dplot(self, px, py):
        """Mesh 2d plot
        Params:
            px, py           [N, 1]

        Output:
            imgs
        """
        N = px.shape[0]

        xv, yv = np.meshgrid(px, py)
        u = np.concatenate([xv.reshape(N, N, 1), yv.reshape(N, N, 1)], axis=2) # [N, N , 2]
        for n in range(N):
            rl = u[n, :, :].reshape(N, 2)
            vl = u[:, n, :].reshape(N, 2)

            plt.plot(rl[:, 0], rl[:, 1], "-g", linewidth=0.5)  # plot row lines
            plt.plot(vl[:, 0], vl[:, 1], "-g", linewidth=0.5)  # plot column lines

    def annotated_img_grid2(self, imgs, iter, width, height, region, name="draw"):
        """Plot the images with position where the picture changes
        --------------------------------------------
        iter              i th iteration
        x                 Shape:(batch_size, x_dims)
        width, height     Figure size
        region            [batch_size, Nfilter, 2]   [center_x, center_y]
        ------
        """
        plt.close("all")
        batch_size = imgs.shape[0]
        Nimg = int(np.sqrt(batch_size))
        imgs = imgs[:Nimg**2, :].reshape([Nimg, Nimg, -1])
        region = region[:Nimg**2].reshape([Nimg, Nimg, -1, 2])

        self.fast_animated_grid(imgs, region, height, width,
                                path="./pick", name="%s_%s" % (name, str(iter).zfill(2)), mode="distributed")


    def fast_animated_grid(self, x, r, height, width, path="./pick", name="", mode="spatial"):
        """Create a canvas to picture all the example  very quickly
           x,image     [niter, batch size, x_dim] -> [height, width]
           r region    [niter, batch_size, r_dim] (for spatial) or [niter, batch_size, N, 2] (for distributed)
        """
        padsize = 1
        padval = 0.5
        ph = height + 2 * padsize
        pw = width + 2 * padsize

        niter = x.shape[0]
        batch_size = x.shape[1]

        img = np.ones((batch_size * ph, niter * pw)) * padval
        pats = []
        for i in range(niter):
            for j in range(batch_size):
                row_start = i * pw + padsize
                row_end = row_start + width
                col_start = j * ph + padsize
                col_end = col_start + height
                img[col_start:col_end, row_start:row_end] = self.scale_norm(x[i, j, :]).reshape(height, width)

                if mode == "spatial":
                    pat = patches.Rectangle([r[i,j, 0]+row_start, r[i, j, 1]+col_start],
                                                r[i, j, 2], r[i, j, 3], color="r", fill=False)
                    pats.append(pat)

                elif mode == "distributed":
                    px, py = r[i, j, :, 0]+row_start, r[i, j, :, 1]+col_start
                    Nf = len(px)
                    xv, yv = np.meshgrid(px, py)
                    u = np.concatenate([xv.reshape(Nf, Nf, 1), yv.reshape(Nf, Nf, 1)], axis=2)  # [N, N , 2]
                    pats.append(u)

        plt.close('all')
        nb = niter* pw/(batch_size * ph)
        fig = plt.figure(figsize=(9*nb, 9), dpi=500)   # (width, height)

        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img, cmap=plt.cm.gray)
        if mode == "spatial":
            for pat in pats:
                ax.add_patch(pat)

        elif mode == "distributed":
            for pat in pats:
                for n in range(Nf):
                    rl = pat[n, :, :].reshape([Nf, 2])
                    vl = pat[:, n, :].reshape([Nf, 2])

                    ax.plot(rl[:, 0], rl[:, 1], "-g", linewidth=1)  # plot row lines
                    ax.plot(vl[:, 0], vl[:, 1], "-g", linewidth=1)  # plot column lines

        ax.set_axis_off()
        ax.autoscale_view(scalex=True, scaley=True)

        ax.margins(0, 0)
        plt.tight_layout(h_pad=0.1, w_pad=0.2)
        plt.savefig(path + "/%s.png" % name)

    def attention_1D_show(self,iter, l, reg_rgs, state_pgs, reg_wpgs, path="", name=""):
        """Visualize the attention position
            l        :           (batch_size, l_dim)
            reg_rgs              (batch_size, l_dimN(<l_dim))
            iter                  iteration
        """
        plt.close("all")
        batch_size = l.shape[0]
        Nimg = min(int(np.sqrt(batch_size)), 5)
        l, reg_rgs, state_pgs, reg_wpgs = l[:Nimg ** 2], reg_rgs[:Nimg ** 2], \
                                          state_pgs[:Nimg ** 2], reg_wpgs[:Nimg ** 2]

        fig = plt.figure(num=1, figsize=(2 * 8, 8))
        gs = gridspec.GridSpec(Nimg * 2, Nimg)
        ax = []
        xl = np.arange(l.shape[1])
        xs = np.arange(state_pgs.shape[1])

        for n, (lt, rg_r, state, rg_w) in enumerate(zip(l, reg_rgs, state_pgs, reg_wpgs)):
            row = n // Nimg
            col = n % Nimg
            ax.append(
                fig.add_subplot(gs[2 * row, col])
            )
            ax[-1].plot(xl, lt, "#E24A33", lw=1.0)
            ax[-1].fill_between(xl, 0, lt, facecolor='#E24A33', alpha=0.2)
            for yv in rg_r:
                ax[-1].axvline(yv, color="g", linestyle="solid", lw=2, alpha=0.5)
            ax[-1].spines["right"].set_visible(True)
            ax[-1].spines["top"].set_visible(True)
            ax[-1].set_xticklabels([])
            if col != 0:
                ax[-1].set_yticklabels([])
            else:
                ax[-1].tick_params(axis="y", which="major", labelsize=5)

            ax.append(
                fig.add_subplot(gs[2 * row + 1, col]))
            ax[-1].plot(xs, state, '#988ED5', linewidth=1.5)
            ax[-1].fill_between(xs, 0, state, facecolor='#988ED5', alpha=0.2)
            for yv in rg_w:
                ax[-1].axvline(yv, color="g", linestyle="solid", lw=2, alpha=0.5)
            ax[-1].spines["right"].set_visible(True)
            ax[-1].spines["top"].set_visible(True)
            ax[-1].set(xlim=[0, len(xs)-1])
            if row != Nimg - 1:
                ax[-1].set_xticklabels([])
            else:
                ax[-1].tick_params(axis="x", which="major", labelsize=7)
            if col != 0:
                ax[-1].set_yticklabels([])
            else:
                ax[-1].tick_params(axis="y", which="major", labelsize=5)

        # plt.suptitle("iteration_{}".format(iter))
        plt.tight_layout(h_pad=0.2, w_pad=0.2)
        path_pic = os.path.join(path, "%s_%s.png" % (name, str(iter).zfill(2)))
        plt.savefig(path_pic)

    def attention_1n_show(self, l, reg_rgs, state_pgs, reg_wpgs, sample_number=5, path="", name=""):
        """reg_rgs, state_pgs, reg_wpgs                     [niter, batch_size, x_dim]
           sample_number                                    the number of samples
        """
        plt.close("all")
        batch_size = min(reg_rgs.shape[1], sample_number)
        reg_rgs, state_pgs, reg_wpgs = reg_rgs[:, :batch_size, :], \
                                          state_pgs[:, :batch_size, :], reg_wpgs[:, :batch_size, :]
        l = l[:batch_size, :]

        niter = reg_rgs.shape[0]
        xl = np.arange(l.shape[1])
        xs = np.arange(state_pgs.shape[2])

        rate = batch_size/(niter + 1) * 4
        fig = plt.figure(num=1, figsize=(rate * 8, 8))          # width x height
        gs = gridspec.GridSpec(niter + 1, batch_size)
        ax = []
        colors = ['#D98880', '#EC7063', '#AF7AC5', '#7FB3D5', '#76D7C4',
                  '#7DCEA0', "#F9E79F", "#F8C471", "#F0B27A", "#E59866"]

        for col in range(batch_size):
            ax.append(fig.add_subplot(gs[0, col]))
            ax[-1].plot(xl, l[col, :], "#E24A33", lw=1.0)
            ax[-1].fill_between(xl, 0, l[col, :], facecolor='#E24A33', alpha=0.2)
            for row in range(niter):
                for t, yv in enumerate(reg_rgs[row, col]):
                    ax[-1].axvline(yv, color=colors[row], linestyle="solid", lw=2, alpha=0.5)
            ax[-1].set_xticklabels([])
            if col != 0:
                ax[-1].set_yticklabels([])
            else:
                ax[-1].tick_params(axis="y", which="major", labelsize=5)
                ax[-1].set_ylabel("condition", fontsize=8)

        for mcol in range(batch_size):
            for jrow in range(niter):
                ax.append(fig.add_subplot(gs[jrow + 1, mcol]))
                state = state_pgs[jrow, mcol, :]
                ax[-1].plot(xs, state, color=colors[jrow], lw=1.0)
                ax[-1].fill_between(xs, state, facecolor=colors[jrow], alpha=0.2)
                for yv in reg_wpgs[jrow, mcol]:
                    ax[-1].axvline(yv, color=colors[jrow], linestyle="solid", lw=2, alpha=0.5)

                ax[-1].set(xlim=[0, len(xs) - 1])
                if jrow != niter - 1:
                    ax[-1].set_xticklabels([])
                else:
                    ax[-1].tick_params(axis="x", which="major", labelsize=7)
                if mcol != 0:
                    ax[-1].set_yticklabels([])
                else:
                    ax[-1].tick_params(axis="y", which="major", labelsize=5)
                    ax[-1].set_ylabel("iteration {}".format(jrow + 1), fontsize=8)

        plt.tight_layout(h_pad=0.2, w_pad=0.2)
        path_pic = os.path.join(path, "%s.png" % (name))
        plt.savefig(path_pic)
    
    def attention_11_show(self, l, reg_rgs, state_pgs, reg_wpgs, sample_number=5, path="", name=""):
        """reg_rgs, state_pgs, reg_wpgs                     [niter, batch_size, x_dim]
                   sample_number                                    the number of samples
        """
        plt.close("all")
        batch_size = min(reg_rgs.shape[1], sample_number)

        reg_rgs, state_pgs, reg_wpgs = reg_rgs[:, :batch_size, :], \
                                          state_pgs[:, :batch_size, :], reg_wpgs[:, :batch_size, :]
        l2 = l[:batch_size, :]

        niter = reg_rgs.shape[0]
        xl = np.arange(l2.shape[1])
        xs = np.arange(state_pgs.shape[2])

        rate = batch_size / (niter + 1) * 8
        fig = plt.figure(num=1, figsize=(rate * 8, 8))
        gs = gridspec.GridSpec(2, batch_size)
        ax = []
        colors = ['#D98880', '#EC7063', '#AF7AC5', '#7FB3D5', '#76D7C4',
                  '#7DCEA0', "#F9E79F", "#F8C471", "#F0B27A", "#E59866"]

        for col in range(batch_size):
            ax.append(fig.add_subplot(gs[0, col]))
            ax[-1].plot(xl, l2[col, :], "#E24A33", lw=1.0)
            ax[-1].fill_between(xl, 0, l2[col, :], facecolor='#E24A33', alpha=0.2)
            for row in range(niter):
                for t, yv in enumerate(reg_rgs[row, col]):
                    ax[-1].axvline(yv, color=colors[row], linestyle="solid", lw=2, alpha=0.5)
            ax[-1].set_xticklabels([])
            if col != 0:
                ax[-1].set_yticklabels([])
            else:
                ax[-1].tick_params(axis="y", which="major", labelsize=5)
                ax[-1].set_ylabel("Condition", fontsize=10)

        for col in range(batch_size):
            ax.append(fig.add_subplot(gs[1, col]))
            for n in range(niter):
                state = state_pgs[n, col, :]
                ax[-1].plot(xs, state, color=colors[n], lw=1.0)
                # ax[-1].fill_between(xs, state, facecolor=colors[n], alpha=0.2)
                for yv in reg_rgs[n, col, :]:
                    ax[-1].axvline(yv, color=colors[n], linestyle="solid", lw=2, alpha=0.5)

            ax[-1].tick_params(axis="x", which="major", labelsize=7)
            ax[-1].set(xlim=[0, len(xs) - 1])
            label = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "", "", "", "", "", "", "", ""]
            ax[-1].set_xticks(np.arange(len(xs)))
            ax[-1].set_xticklabels(label)

            if col != 0:
                ax[-1].set_yticklabels([])
            else:
                ax[-1].tick_params(axis="y", which="major", labelsize=5)
                ax[-1].set_ylabel("Adjustments", fontsize=10)

        plt.tight_layout(h_pad=0.2, w_pad=0.2)
        path_pic = os.path.join(path, "%s.png" % (name))
        plt.savefig(path_pic)

