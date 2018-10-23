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
#                                                                             # FFB5B8 : pink
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

        # plt.close("all")
        # batch_size = imgs.shape[0]
        # N = int(np.sqrt(batch_size))
        # fig = plt.figure(num=1, figsize=(8, 8))
        # gs = gridspec.GridSpec(N, N, wspace=0.1, hspace=0.1)
        # ax = []
        # for n, img in enumerate(imgs):
        #     row = n // N
        #     col = n % N
        #
        #     ax.append(fig.add_subplot(gs[row, col]))
        #     img = self.scale_norm(img).reshape(height, width)
        #     ax[-1].imshow(img, cmap=plt.cm.gray)
        #     rect = patches.Rectangle(
        #         region[n, 0:2], region[n, 2], region[n, 3], color='r', fill=False)
        #     ax[-1].add_patch(rect)
        #     ax[-1].set_axis_off()
        #
        # plt.savefig("./pick/%s_%s.png" % (name, str(iter).zfill(2)))


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

        # fig = plt.figure(num=1, figsize=(8, 8))
        # gs = gridspec.GridSpec(N, N, wspace=0.1, hspace=0.1)
        # ax = []
        # for n, img in enumerate(imgs):
        #     row = n // N
        #     col = n % N
        #
        #     ax.append(fig.add_subplot(gs[row, col]))
        #     img = self.scale_norm(img).reshape(height, width)
        #     ax[-1].imshow(img, cmap=plt.cm.gray)
        #
        #     r  = region[n, :, :]
        #     px, py = r[:, 0], r[:, 1]
        #     Nf = px.shape[0]
        #     xv, yv = np.meshgrid(px, py)
        #     u = np.concatenate([xv.reshape(Nf, Nf, 1), yv.reshape(Nf, Nf, 1)], axis=2)  # [N, N , 2]
        #     for n in range(Nf):
        #         rl = u[n, :, :].reshape([Nf, 2])
        #         vl = u[:, n, :].reshape([Nf, 2])
        #
        #         ax[-1].plot(rl[:, 0], rl[:, 1], "-g", linewidth=1)  # plot row lines
        #         ax[-1].plot(vl[:, 0], vl[:, 1], "-g", linewidth=1)  # plot column lines
        #     ax[-1].set_axis_off()

        # plt.savefig("./pick/%s_%s.png" % (name, str(iter).zfill(2)))

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
