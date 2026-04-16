# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

the __main__ file that should be run 

"""
import numpy as np
import pickle

import matplotlib.pyplot as plt

from pathlib import Path

from _misc import *

from lmfit import model as lmmodel
from scipy.signal import convolve

from tqdm import tqdm
from backfilt import walkDir, filterImage
from fitting import lm_double_gaus2d,setup_double_2d_gauss_model, fit_double_gauss2d_lm
from perspective import integral_preserving_warp, src_dst_from_known_points, get_dst_layout
from backfilt import get_background


def generate_stats(export_path, src, dst, backgroundData=None):
    fmodel = setup_double_2d_gauss_model()

    tifFiles = walkDir(settings.targetDir)
    
    if settings.shortlist is not None:
        filtered = []
        for file in tifFiles:
            for item in settings.shortlist:
                if item in str(file):
                    filtered.append(file)
        tifFiles = filtered

    kernel = eval(settings.kernel)

    stats = []

    vprint(f"Found {len(tifFiles)} Tiff Files")

    stats_path = export_path / f"stats"

    for file in tqdm(tifFiles[settings.start : settings.stop : settings.decimate]):
        name = file.stem
        print(name)
        savefile = export_path / f"{name}_data"

        pixelData = np.array(PIL.Image.open(file))

        if savefile.exists() and not settings.overwrite:
            print(f"found fitresults for {savefile}")
            result = lmmodel.load_modelresult(
                str(savefile), {"lm_double_gaus2d": lm_double_gaus2d}
            )

            x2 = result.userkws["x"]
            y2 = result.userkws["y"]

        else:
            vprint(f"using {len(settings.filters)} x-ray filters")
            for f in settings.filters:
                pixel_data = filterImage(pixelData, f)
            vprint(f"convolving with {settings.kernel}")
            pixelData = convolve(pixelData, kernel, mode="same")

            if backgroundData is not None:
                vprint(f"Subtracting Backgrounds")
                if not backgroundData.shape == pixelData.shape:
                    print("Background missmatch")
                else:
                    A = pixelData.sum()
                    B = backgroundData.sum()
                    pixelData = np.subtract(
                        pixelData,
                        np.multiply(
                            backgroundData, (settings.background_scale * A / B)
                        ),
                    )

            noise_scale = np.percentile(
                pixelData[30:-1, 1:-35], settings.background_clip
            )
            pixelData = np.clip(pixelData - noise_scale, 0, np.inf)

            # ── Canvas dimensions and beam axis ──────────────────────────────────────
            dst_size, dst_offset = get_dst_layout(pixelData.shape, src, dst, settings.dst_padding, settings.resolution)
            
        
            # ── Apply warp with Jacobian correction ───────────────────────────────────
            transformed, axis, _ = integral_preserving_warp(
                pixelData, dst_size, dst_offset, src, dst
            )

            zoom_x_lims = [
                max(0, int(dst_offset[1] - (settings.zoom_radius * settings.resolution))),
                int(dst_offset[1] + (settings.zoom_radius * settings.resolution)),
            ]
            zoom_y_lims = [
                int(dst_offset[0] + (settings.zoom_radius * settings.resolution)),
                max(0, int(dst_offset[0] - (settings.zoom_radius * settings.resolution))),
            ]

            roi = np.array(
                transformed[
                    zoom_x_lims[0] : zoom_x_lims[1], zoom_y_lims[1] : zoom_y_lims[0]
                ]
            )

            if settings.plotBackgroundSubtraction:
                roi_pre = roi

            #if mask is not None:
            #    continue ## TODO: 

            for f in settings.filters:
                roi = filterImage(roi, f)

            if np.max(roi) > settings.ignore_ptvs_below * np.mean(roi):

                if settings.plotBackgroundSubtraction:
                    cbpad = 0
                    cbscale = 0.6
                    rfig, rax = plt.subplots(1, 2, figsize=(12, 6))
                    im1 = rax[0].imshow(roi_pre, vmin=np.min(roi), vmax=np.max(roi))
                    cax1 = rfig.colorbar(
                        im1, ax=rax[0], pad=cbpad, shrink=cbscale, location="right"
                    )

                    im2 = rax[1].imshow(roi, vmin=np.min(roi), vmax=np.max(roi))
                    cax2 = rfig.colorbar(
                        im2, ax=rax[1], pad=cbpad, shrink=cbscale, location="right"
                    )

                x = np.linspace(
                    -settings.zoom_radius, settings.zoom_radius, roi.shape[1]
                )
                y = np.linspace(
                    -settings.zoom_radius, settings.zoom_radius, roi.shape[0]
                )

                x2, y2 = np.meshgrid(x, y)

                result = fit_double_gauss2d_lm(x2, y2, roi, fmodel)
                best_values = result.best_values
            else:
                print(
                    "It looks like is no signal in this image: {}".format(file)
                )
                print("max = {}".format(np.max(roi)))
                print("mean = {}".format(np.mean(roi)))
                print("med = {}".format(np.median(roi)))
                result = None
                x2 = None
                y2 = None
                roi = None

        if result is not None:

            name = file.stem
            saveplot = export_path / f"{name}_plot"

            fitted = fmodel.func(x2, y2, **result.best_values)

            stats.append([result.rsquared, result.best_values, result.covar])

            if saveplot.exists() and not settings.overwrite:
                 pass
            else:
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                    im = ax.imshow(
                        roi,
                        cmap=plt.cm.jet,
                        origin="lower",
                        extent=(x.min(), x.max(), y.min(), y.max()),
                    )
                    ax.contour(
                        x2,
                        y2,
                        fitted,
                        4,
                        colors="black",
                        extent=(x.min(), x.max(), y.min(), y.max()),
                        linewidths=0.3,
                    )

                    fig.colorbar(im, ax=ax)

                    V_C = (
                        2
                        * np.pi
                        * result.best_values["amplitude_1"]
                        * result.best_values["sigma_x_1"]
                        * result.best_values["sigma_y_1"]
                    )
                    V_B = (
                        2
                        * np.pi
                        * result.best_values["amplitude_2"]
                        * result.best_values["sigma_x_2"]
                        * result.best_values["sigma_y_2"]
                    )

                    ax.set_title(
                        f"\n BUNCH: [A:{result.best_values['amplitude_2']:.1f},"
                        + r" $\sigma_x$"
                        + f":{result.best_values['sigma_x_2']:.2f},"
                        + r" $\sigma_y$"
                        + f":{result.best_values['sigma_y_2']:.2f}] \n Integral {V_B:.0f} at ["
                        + r"$\theta_x$"
                        + f":{result.best_values['xo_1']:.1f},"
                        + r"$\theta_y$"
                        + f":{result.best_values['yo_1']:.1f}]"
                        + f"\n CLOUD: [A:{result.best_values['amplitude_1']:.1f},"
                        + r" $\sigma_x$"
                        + f":{result.best_values['sigma_x_1']:.2f},"
                        + r" $\sigma_y$"
                        + f":{result.best_values['sigma_y_1']:.2f}] \n Integral {V_C:.0f} at ["
                        + r"$\theta_x$"
                        + f":{result.best_values['xo_2']:.1f},"
                        + r"$\theta_y$"
                        + f":{result.best_values['yo_2']:.1f}]"
                    )

                    ax.set_xlabel(r"$\theta_x$")
                    ax.set_ylabel(r"$\theta_y$")

                    fig.tight_layout()

                    if settings.saving:
                        lmmodel.save_modelresult(result, str(savefile))
                        fig.savefig(str(saveplot), dpi=600)
                        plt.close(fig)
                    else:
                        fig.show()
                        settings.blockingPlot = True
                except NameError as e:
                    print(
                        f"raised Name Error {e}, when generating {file}, this happens when data exists but plot does not"
                    )
    if settings.saving:
        with open(stats_path, "wb") as handle:
             pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stats

def generate_report(stats, export_path):
    report_figures = []
    ##
    #   Pointing:
    #
    u_x = np.array([stats[i][1]["xo_2"] for i in range(len(stats))])
    u_y = np.array([stats[i][1]["yo_2"] for i in range(len(stats))])

    report_figures.append([plt.figure(figsize=(9, 9), dpi=360), "pointing"])
    report_figures[-1][0].set_tight_layout(True)

    gs_hist = report_figures[-1][0].add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    report_figures[-1][0].set_size_inches(9, 9)

    nbins = int(max(np.sqrt(len(stats)), 15))
    pAx = report_figures[-1][0].add_subplot(
        gs_hist[1, 0],
    )

    pAx.minorticks_on()
    uux = np.mean(u_x)
    uuy = np.mean(u_y)

    u_0_x = u_x - uux
    u_0_y = u_y - uuy
    pAx.hist2d(
        u_0_x,
        u_0_y,
        nbins,
        [
            [-settings.zoom_radius, settings.zoom_radius],
            [-settings.zoom_radius, settings.zoom_radius],
        ],
    )
    pAx.set_aspect("equal")
    # pAx.add_patch(Circle((0,0),6,ec='red',fill=False))

    xlabels = [
        item.get_text() if item.get_text() != "0" else r"$\mu_x$"
        for item in pAx.get_xticklabels()
    ]
    pAx.set_xticklabels(xlabels)

    ylabels = [
        item.get_text() if item.get_text() != "0" else r"$\mu_y$"
        for item in pAx.get_yticklabels()
    ]
    pAx.set_yticklabels(ylabels)

    pax_histx = report_figures[-1][0].add_subplot(gs_hist[0, 0], sharex=pAx)
    pax_histy = report_figures[-1][0].add_subplot(gs_hist[1, 1], sharey=pAx)

    pax_text = report_figures[-1][0].add_subplot(gs_hist[0, 1])
    pax_text.axis("off")  # Turn off axis for the text subplot

    pax_histx.axes.get_xaxis().set_visible(False)
    pax_histy.axes.get_yaxis().set_visible(False)

    xcount, xbins, _ = pax_histx.hist(
        u_0_x,
        nbins,
        density=True,
        color="green",
    )

    xbins = xbins[:-1] + (xbins[1] - xbins[0]) / 2

    ycount, ybins, _ = pax_histy.hist(
        u_0_y, nbins, density=True, color="green", orientation="horizontal"
    )
    ybins = ybins[:-1] + (ybins[1] - ybins[0]) / 2
    xmin, xmax = pax_histx.get_xlim()
    ymin, ymax = pax_histy.get_ylim()

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    def gauss(x, A, s):
        return A * np.exp(-((x / s) ** 2))

    try:
        poptx, pcovx = curve_fit(gauss, xbins, xcount, maxfev=6000)
    except RuntimeError as E:
        print("WARNING", E)
        popt_x = [np.sqrt(2 * np.pi * u_x.var()), np.sqrt(u_x.var())]
    try:
        popty, pcovy = curve_fit(gauss, ybins, ycount)
    except RuntimeError as E:
        print("WARNING", E)
        popt_x = [np.sqrt(2 * np.pi * u_y.var()), np.sqrt(u_y.var())]

    px = gauss(x, *poptx)
    py = gauss(y, *popty)

    pax_histx.plot(x, px, "k", linewidth=2)
    pax_histy.plot(py, y, "k", linewidth=2)

    xlab = r"$\theta_x$ [mRad]"
    ylab = r"$\theta_y$ [mRad]"

    rep_str = "$\\mu_x$ = {:.2f},\n$\\sigma_x$ = {:.2f}\n\n$\\mu_y$ = {:.2f},\n$\\sigma_y$ = {:.2f}".format(
        *poptx, *popty
    )
    pax_text.text(
        0.1, 0.6, rep_str, fontsize=10, verticalalignment="top", family="monospace"
    )

    pAx.set_xlabel(xlab)
    pAx.set_ylabel(ylab)
    report_figures[-1][0].set_tight_layout(True)

    ##
    #   Bunch Emittence:
    #
    s_x = np.array([stats[i][1]["sigma_x_2"] for i in range(len(stats))])
    s_y = np.array([stats[i][1]["sigma_y_2"] for i in range(len(stats))])
    th = np.array([stats[i][1]["theta_2"] for i in range(len(stats))])

    report_figures.append([plt.figure(figsize=(9, 12)), "emittence_b"])
    maj_ax = report_figures[-1][0].add_subplot(311)
    min_ax = report_figures[-1][0].add_subplot(312)
    th_ax = report_figures[-1][0].add_subplot(313)
    report_figures[-1][0].suptitle("Analysis of the Bunch Sizes")
    report_figures[-1][0].set_tight_layout(True)

    maj_ax.hist(s_x, nbins)
    mxaxlab = "$\sigma_{major}$" + "[mRad] mean:{:.2f}  s.d.:{:.2f}".format(
        s_x.mean(), np.sqrt(s_x.var())
    )
    maj_ax.set_xlabel(mxaxlab)
    min_ax.hist(s_y, nbins)
    minxaxlab = "$\sigma_{minor}$" + "[mRad] mean:{:.2f}  s.d:{:.2f}".format(
        s_y.mean(), np.sqrt(s_y.var())
    )
    min_ax.set_xlabel(minxaxlab)

    th = ((th + 90) % 180) - 90  # Center around 0
    th_ax.hist(th, nbins)
    th_ax.set_xlabel(
        "$\phi [^\circ]$ (from x to major axis) mean:{:.2f}  s.d.:{:.2f}".format(
            th.mean(), np.sqrt(th.var())
        )
    )

    ##
    #   Bunch Twist:
    #

    report_figures.append((plt.figure(figsize=(9, 12)), "twist"))
    th_scat3d_ax = report_figures[-1][0].add_subplot(projection="3d")
    th_scat3d_ax.scatter(s_x, s_y, th)
    th_scat3d_ax.set_xlabel("Major")
    th_scat3d_ax.set_ylabel("Minor")
    th_scat3d_ax.set_zlabel("Theta")

    ##
    #   Cloud Emittence:
    #
    s_x = np.log10(np.array([stats[i][1]["sigma_x_1"] for i in range(len(stats))]))
    s_y = np.log10(np.array([stats[i][1]["sigma_y_1"] for i in range(len(stats))]))
    th = np.array([stats[i][1]["theta_1"] for i in range(len(stats))])

    report_figures.append([plt.figure(figsize=(9, 12)), "emittence_c"])
    maj_ax = report_figures[-1][0].add_subplot(311)
    min_ax = report_figures[-1][0].add_subplot(312)
    th_ax = report_figures[-1][0].add_subplot(313)
    report_figures[-1][0].suptitle("Analysis of the Cloud Sizes")
    report_figures[-1][0].set_tight_layout(True)

    maj_ax.hist(s_x, nbins)
    mxaxlab = (
        "$log_{10}(\sigma_{major})$"
        + "[log(mRad)] mean:{:.2f}  s.d.:{:.2f}".format(s_x.mean(), np.sqrt(s_x.var()))
    )
    maj_ax.set_xlabel(mxaxlab)
    min_ax.hist(s_y, nbins)
    minxaxlab = (
        "$log_{10}(\sigma_{minor})$"
        + "[log(mRad)] mean:{:.2f}  s.d:{:.2f}".format(s_y.mean(), np.sqrt(s_y.var()))
    )
    min_ax.set_xlabel(minxaxlab)

    th = ((th + 90) % 180) - 90  # Center around 0
    th_ax.hist(th, nbins)
    th_ax.set_xlabel(
        "$\phi [^\circ]$ (from x to major axis) mean:{:.2f}  s.d.:{:.2f}".format(
            th.mean(), np.sqrt(th.var())
        )
    )

    ##
    #   Bunch Amplitude and Contrast:
    #
    def gaussian_volume(A, S_x, S_y):
        return A * S_x * S_y * 2 * np.pi

    ratio = []
    V_1 = []
    V_2 = []
    for i, shot in enumerate(stats):
        V1 = gaussian_volume(
            shot[1]["amplitude_1"], shot[1]["sigma_x_1"], shot[1]["sigma_y_1"]
        )
        V2 = gaussian_volume(
            shot[1]["amplitude_2"], shot[1]["sigma_x_2"], shot[1]["sigma_y_2"]
        )
        if V1 > 0 and V2 > 0:
            V_1.append(np.log10(V1))
            V_2.append(np.log10(V2))
            if V1 / V2 < 100:
                ratio.append(V1 / V2)

    report_figures.append((plt.figure(figsize=(9, 12)), "Charge Ratio"))

    V1ax = report_figures[-1][0].add_subplot(3, 1, 1)
    V1ax.hist(V_1, bins=30)
    V1ax.set_title("Log_10 of the volume under the wider gaussian g_1")

    V2ax = report_figures[-1][0].add_subplot(3, 1, 2)
    V2ax.hist(V_2, bins=30)
    V2ax.set_title("Log_10 of the volume under the thinner gaussian g_2")

    Rax = report_figures[-1][0].add_subplot(3, 1, 3)
    Rax.hist(ratio, bins=30)
    # Rax.set_xlim(0,100)
    Rax.set_title("ratio of Volumes where less than 500")

    for fig in report_figures:
        if settings.saving:
            fig_path = export_path / f"{fig[1]}_fig"
            fig[0].savefig(fig_path, dpi=600)
        else:
            fig[0].show()
            settings.blockingPlot = True

def main(input_deck_path=None):
    
    settings.update_user_settings(input_deck_path=input_deck_path)
    
    if settings.assert_reasonable():
        export_path = Path(settings.targetDir) / "EXPORTED"  # name the subdirectory to export to

        export_path.mkdir(parents=True, exist_ok=True)

        src, dst = src_dst_from_known_points(settings.known_points,
                                                    settings.units,
                                                    settings.resolution,
                                                    settings.lanex_onAx_dist, 
                                                    settings.lanex_theta, 
                                                    settings.lanex_inPlane_dist, 
                                                    settings.lanex_height,
                                                    settings.lanex_vertical_offset)

        backgroundData = get_background()

        stats_pickle = export_path / "stats.pickle"
        stats_npy = export_path / "stats.npy"

        if stats_pickle.exists() and not settings.overwrite:
            print("found existing pikle stats file in export directory")
            
            with open(stats_pickle, 'rb') as handle:
                 stats = pickle.load(handle)
        elif stats_npy.exists() and not settings.overwrite:
            print("found existing numpy stats file in export directory")

            stats = np.load(stats_npy, allow_pickle=True)

        else:
            stats = generate_stats(export_path, src, dst, backgroundData)

        report = generate_report(stats, export_path)
        
        if settings.blockingPlot:
            plt.show()
            input("close? : ")
  
if __name__ == "__main__":
    main()
