#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:03:05 2019

@author: maddy
"""

import numpy as np
import matplotlib.pyplot as plt
from expyfun.io import reconstruct_dealer, read_tab_raw, read_hdf5
import ast
import scipy.stats as ss
from scipy.optimize import fmin
from expyfun.analyze import sigmoid
from scipy.stats import linregress, ttest_rel, ttest_ind, ttest_1samp, wilcoxon
import seaborn.apionly as sns
from pandas import DataFrame
from scipy.io import savemat
from expyfun.io import write_hdf5, read_hdf5

import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
subjects = np.array(['001', '002', '003', '005', '006', '007', '008', '009', 
                     '010', '011', '012', '014', '015', '016', '017', '018',
                     '019', '020', '021', '022'])
conditions = np.array(['control', 'matched'])
durations = np.array([0.1, 0.3, 1.])

thresholds = np.zeros((len(subjects), 2, 3))
pvals_rev = np.zeros((len(subjects), 3, 10))

for i, subject in enumerate(subjects):
    data = read_hdf5(subject + '.hdf5')
    thresholds[i] = data['thresh']
    pvals_rev[i] = data['pval']
    
    
effect = -np.diff(np.log(thresholds), 1, 1).squeeze()
control_thresholds = np.log(thresholds[:, 0, :].squeeze())
match_thresholds = np.log(thresholds[:, 1, :].squeeze())
data = read_hdf5('original_diff.hdf5')
cont = data['thresh']
diffs = data['diffs']
# %%
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.patches import Rectangle

fontsize=9
rc('text', usetex=False)
rcParams['font.sans-serif'] = "Arial"
plt.rc('font', size=fontsize)
plt.rc('axes', titlesize=fontsize)
plt.rc('xtick', labelsize=fontsize-2)
plt.rc('ytick', labelsize=fontsize-2)
titles = ['100 ms', '300 ms', '1000 ms']
lw = 1.25
ms = 4
import statsmodels.api as sm
from scipy.stats import shapiro, pearsonr
for i in np.arange(3):
    plt.figure(i, figsize=(3.5, 1.9), dpi=600)
    data = DataFrame(data={'control': control_thresholds[:, i].ravel(), 'matched': match_thresholds[:, i].ravel()})
#    sns.regplot('control', 'matched', data, ci=95, marker='o', color='C0', line_kws={'lw':lw})
    plt.plot(np.exp(control_thresholds[:, i]), np.exp(match_thresholds[:, i]), 'o', c='C0', ms=ms)
    for ii in np.arange(len(subjects)):
        p = pvals_rev[ii, i, 0]
        if p < 0.025 or p > 0.975:
            plt.plot(np.exp(control_thresholds[ii, i]), np.exp(match_thresholds[ii, i]), 'o', mfc='C0', mec='C0', ms=ms)
        else:
            plt.plot(np.exp(control_thresholds[ii, i]), np.exp(match_thresholds[ii, i]), 'o', mfc='w', mec='C0', ms=ms)
    plt.plot([0, 25], [0, 25], 'k', lw=(lw-.25), zorder=-100)
    plt.title(titles[i])
#    slope, intercept, r_value, p_value, std_err = linregress(control_thresholds[:, i][~np.isnan(control_thresholds[:, i])], match_thresholds[:, i][~np.isnan(match_thresholds[:, i])])
#    print([slope, intercept, r_value, p_value])
    x_fit = control_thresholds[:, i][~np.isnan(control_thresholds[:, i])]
    y_fit = match_thresholds[:, i][~np.isnan(match_thresholds[:, i])]
    print(i, ttest_rel(x_fit, y_fit))
    p, res, _, _, _ = np.polyfit(np.zeros(x_fit.shape), y_fit-x_fit, 0, full=True)
    r2 = 1 - np.sum((y_fit- x_fit - p)**2)/np.sum((y_fit - np.mean(y_fit))**2)
    print(i, p, r2)
    x_fit = sm.add_constant(x_fit)
    mod = sm.OLS(y_fit, x_fit)
    reg = mod.fit()
    print(mod.fit().summary())
    params = mod.fit().params
    x_plot = np.arange(0, 25, .1)
    plt.plot(x_plot, np.exp(params[0] + params[1] * np.log(x_plot)), 'C0')
    plt.plot(x_plot, np.exp(np.polyval(p, np.log(x_plot)) + np.log(x_plot)), 'C1')
    conf = mod.fit().conf_int()
    print('R2', mod.fit().rsquared)
    rect = Rectangle((.25,16.5),16,10.5,linewidth=.25,edgecolor='k',facecolor='none')
#    plt.gca().add_patch(rect)
#    plt.show()
    [plt.text(.5, [24, 20.5, 17][i], ['y-int:', 'slope:', u'R²:'][i]) for i in np.arange(3)]
    [plt.text(4.5, [24, 20.5][i], str(params[i])[:4] + '   [' + str(conf[i][0])[:4] + ', ') if np.abs(conf[i][0]) > 0.1 else plt.text(4.5, [24, 20.5][i], str(params[i])[:4] + '   [' + '{:.0e}'.format(conf[i][0])[:4] + '{:.0e}'.format(conf[i][0])[-1] + ', ') for i in np.arange(2)]
    [plt.text(11.5, [24, 20.5][i], str(conf[i][1])[:4] + ']') for i in np.arange(2)]
#    plt.xlabel(u'Central Threshold (°)')
    plt.text(4.5, 17, str(mod.fit().rsquared)[:4])
    plt.text(4.5, 14, str(r2)[:4], color='C1')
    plt.ylabel(u'Matched Threshold (°)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim([0, 27])
    plt.ylim([0, 27])
    plt.tight_layout()
    plt.savefig('cent_match_fits{}.pdf'.format(int(i)), dpi=600)
    resid = mod.fit().resid
    print(['normal?', shapiro(resid)])
    print(['correlated?', pearsonr(x_fit[:, 1], np.abs(resid))])
    plt.figure(6)
    plt.plot(x_fit, np.abs(resid), 'o')
plt.figure(3, figsize=(3.5, 1.9), dpi=600)
data = DataFrame(data={'control': cont, 'matched': (cont - diffs.squeeze())})
#sns.regplot('control', 'matched', data, ci=95, marker='o', color='C0', line_kws={'lw':lw})
plt.plot(cont, cont - diffs.squeeze(), '+', c='C0', ms=ms)
x_fit = np.log(cont)
y_fit = np.log(cont - diffs.squeeze())
print('old', ttest_rel(x_fit, y_fit))
p, res, _, _, _ = np.polyfit(np.zeros(x_fit.shape), y_fit-x_fit, 0, full=True)    
r2 = 1 - res/np.sum((y_fit - np.mean(y_fit))**2)
print('old', p, res)
x_fit = sm.add_constant(x_fit)
mod_o = sm.OLS(y_fit, x_fit)
print(mod_o.fit().summary())
params = mod_o.fit().params
x_plot = np.arange(0, 25, .1)
plt.plot(x_plot, np.exp(params[0] + params[1] * np.log(x_plot)), 'C0')
plt.plot(x_plot, np.exp(np.polyval(p, np.log(x_plot)) + np.log(x_plot)), 'C1')
plt.plot([0, 25], [0, 25], 'k', lw=(lw-.25), zorder=-100)
conf = mod_o.fit().conf_int()
print('R2', mod_o.fit().rsquared)
[plt.text(.5, [24, 20.5, 17][i], ['y-int:', 'slope:', u'R²:'][i]) for i in np.arange(3)]
[plt.text(4.5, [24, 20.5][i], str(params[i])[:4] + '   [' + str(conf[i][0])[:4] + ', ') if np.abs(conf[i][0]) > 0.1 else plt.text(4.5, [24, 20.5][i], str(params[i])[:4] + '   [' + '{:.0e}'.format(conf[i][0])[:4] + '{:.0e}'.format(conf[i][0])[-1] + ', ') for i in np.arange(2)]
[plt.text(11.5, [24, 20.5][i], str(conf[i][1])[:4] + ']') for i in np.arange(2)]
plt.text(4.5, 17, str(mod_o.fit().rsquared)[:4])
plt.text(4.5, 14, str(r2)[:4], color='C1')
rect = Rectangle((.25,16.5),16,10.5,linewidth=.25,edgecolor='k',facecolor='none')
resid = mod_o.fit().resid
print(['normal?', shapiro(resid)])
print(['correlated?', pearsonr(x_fit[:, 1], np.abs(resid))])
#plt.gca().add_patch(rect)
#plt.show()
plt.title('Original Experiment (300 ms)')
plt.xlabel(u'Central Threshold (°)')
plt.ylabel(u'Matched Threshold (°)')
plt.xlim([0, 27])
plt.ylim([0, 27])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('cent_match_fits_orig.pdf', dpi=600)
subs = np.array([6 * [i] for i in np.arange(20)], dtype=int).ravel()
durs = np.array(40 * [100, 300, 1000], dtype=int)
condition = np.array(30 * [0, 0, 1, 1], dtype=int).ravel()
thresholds_2 = thresholds.ravel()

plt.figure(6)
plt.plot(x_fit[:, 1], resid, 'o')
plt.legend(('100', '300', '1000', 'orig'))

plt.figure(4)

x_fit = control_thresholds.ravel()[~np.isnan(control_thresholds.ravel())]
y_fit = match_thresholds.ravel()[~np.isnan(match_thresholds.ravel())]
plt.plot(x_fit, y_fit, 'o', c='C0')
x_fit = sm.add_constant(x_fit)
mod_o = sm.OLS(y_fit, x_fit)
print(mod_o.fit().summary())
params = mod_o.fit().params
x_plot = np.arange(0, 25, .1)
plt.plot(x_plot, params[0] + params[1] * x_plot, 'C0')
plt.plot([0, 25], [0, 25], 'k', lw=(lw-.25), zorder=-100)
[plt.text(0, [24, 19][i], ['y-int: ', 'slope: '][i] + str(params[i])[:5] + '   [' + str(conf[i][0])[:5] + ',  ' + str(conf[i][1])[:5] + ']') for i in np.arange(2)]
plt.title('Pooled durations')

subs = np.array([3 * [i] for i in np.arange(20)], dtype=int).ravel()
durs = np.array(20 * [100, 300, 1000], dtype=int)
#condition = np.array(30 * [0, 0, 1, 1], dtype=int).ravel()
thresholds_log = np.log(thresholds_2)
#data_array = dict(Subject=subs,
#                  Vis=condition,
#                  Duration=durs,
#                  Effect=np.log(thresholds_2))
#formula = "Effect ~ C(Duration) + C(Subject)"
#formula = "Effect ~ Duration + C(Vis) + Duration:C(Vis)"
#df = pd.DataFrame(data_array)
#
#model = ols(formula, df).fit()
#aov_table = anova_lm(model, typ=2, robust='hc3')
#print(aov_table)
formula = "Matched ~ Central + C(Duration) + Central:C(Duration)"
which = ~np.isnan(control_thresholds.ravel() + match_thresholds.ravel())
data_array = dict(Subject=subs[which],
                  Central=control_thresholds.ravel()[which],
                  Duration=durs[which],
                  Matched=match_thresholds.ravel()[which])
df = pd.DataFrame(data_array)
md = smf.mixedlm(formula, df, groups=df["Subject"])
mdf = md.fit()
print(mdf.summary())
#thresholds = thresholds.ravel()
stop
# %%
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
fontsize=9

ms=3
lw=1.25
markers = ['o', 'o', 'o']
colors = ['C0', 'C0', 'C0']

plt.close('all')
[plt.figure(figsize=(3, 1.75), dpi=600) for _ in np.arange(3)]
for e, co, pv in zip(effect, control_thresholds, pvals_rev[:, :, 0]):
    for c, ee, marker, col, p, i, ax in zip(co, e, markers, colors, pv, [1, 2, 3], [0, 1, 2]):
        plt.figure(i)
        plt.plot(c, ee, marker=marker, c=col, ms=ms)
        plt.plot(c, ee, marker=marker, c='w', ms=ms-1.5)
        if p < 0.025 or p > 0.975:
            plt.plot(c, ee, marker=marker, c=col, ms=ms)
data1 = DataFrame(data={'control_thresholds': control_thresholds[:, 0].ravel(), 'effect': effect[:, 0].ravel()})
plt.figure(1)
err = sns.regplot('control_thresholds', 'effect', data1, ci=95, marker='', color=colors[0], line_kws={'lw':lw})
plt.plot([0, 25], [0, 0], 'k', lw=(lw-.25), zorder=-100)
plt.title('100 ms')
plt.xlabel(' ')
plt.ylabel(u'\nThreshold\nimprovement (ln °)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([-6, 12])
plt.xlim([0, 25])
plt.ylim([-1, 1])
plt.text(2, 8, 'r = 0.31 \np = 0.34', fontsize=fontsize, color=colors[0])
plt.tight_layout()
plt.savefig('paper1.pdf', dpi=600)

plt.figure(2)
data2 = DataFrame(data={'control_thresholds': control_thresholds[:, 1].ravel(), 'effect': effect[:, 1].ravel()})
err = sns.regplot('control_thresholds', 'effect', data2, ci=95, marker='', color=colors[1], line_kws={'lw':lw})
plt.plot([0, 25], [0, 0], 'k', lw=lw-.25, zorder=-100)
plt.title('300 ms')
plt.xlabel(' ')
plt.ylabel(u'\nThreshold\nimprovement (ln °)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([-6, 12])
plt.xlim([0, 25])
plt.ylim([-1, 1])
plt.text(19, -5.3, 'r = 0.50 \np = 3e-2', fontsize=fontsize, color=colors[1])
plt.tight_layout()
plt.savefig('paper2.pdf', dpi=600)

plt.figure(3)
data3 = DataFrame(data={'control_thresholds': control_thresholds[:, 2].ravel(), 'effect': effect[:, 2].ravel()})
err = sns.regplot('control_thresholds', 'effect', data3, ci=95, marker='', color=colors[2], line_kws={'lw':lw})
plt.plot([0, 25], [0, 0], 'k', lw=lw-.25, zorder=-100)
plt.title('1000 ms')
plt.xlabel(' ')
plt.ylabel(u'\nThreshold\nimprovement (ln °)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim([0, 25])
plt.ylim([-6, 12])
plt.ylim([-1, 1])
plt.text(19, -5.3, 'r = 0.19 \np = 0.45', fontsize=fontsize, color=colors[2])
plt.tight_layout()
plt.savefig('paper3.pdf', dpi=600)

#plt.tight_layout()
#plt.savefig('breakout_lin_fits.pdf', dpi=600)
# %%
fig = plt.figure(figsize=(.5, 1.75))
kernel = gaussian_kde(effect[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())], 0.3)
plt.fill_betweenx(np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], 
                  10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[0], zorder=101)
plt.plot([0, 3.5], [0, 0], 'w', lw=lw-.25, zorder=102)
plt.ylim([-6, 12])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xlabel(' ')
plt.tight_layout()
plt.subplots_adjust(left=0, right=0.98, top=0.81, bottom=0.302)
plt.savefig('breakout_marginals_1.pdf', dpi=600)
fig = plt.figure(figsize=(.5, 1.75))
kernel = gaussian_kde(effect[:, 1].ravel()[~np.isnan(effect[:, 1].ravel())], 0.3)
plt.fill_betweenx(np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], 
                  10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[1], zorder=100)
plt.plot([0, 3.5], [0, 0], 'w', lw=lw-.25, zorder=100)
plt.ylim([-6, 12])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xlabel(' ')
plt.tight_layout()
plt.subplots_adjust(left=0, right=0.98, top=0.81, bottom=0.302)
plt.savefig('breakout_marginals_2.pdf', dpi=600)
fig = plt.figure(figsize=(.5, 1.75))
kernel = gaussian_kde(effect[:, 2].ravel()[~np.isnan(effect[:, 2].ravel())], 0.3)
plt.fill_betweenx(np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], 
                  10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[2], zorder=99)
plt.plot([0, 3.5], [0, 0], 'w', lw=lw-.25, zorder=100)
plt.ylim([-6, 12])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xlabel(' ')
plt.tight_layout()
plt.subplots_adjust(left=0, right=0.98, top=0.81, bottom=0.302)
plt.savefig('breakout_marginals_3.pdf', dpi=600)
# %%
plt.figure(figsize=(3, 1.75), dpi=600)

for e, co, pv in zip(effect, control_thresholds, pvals_rev[:, :, 0]):
    plt.plot(co, e, 'o', c='C0', ms=ms)
    plt.plot(co, e, 'o', c='w', ms=ms-1.5)
    plt.plot(co[pv < 0.025], e[pv < 0.025], 'o', c='C0', ms=ms)
    plt.plot(co[pv > 0.975], e[pv > 0.975], 'o', c='C0', ms=ms)
data_all = DataFrame(data={'control_thresholds': control_thresholds.ravel(), 'effect': effect.ravel()})
err = sns.regplot('control_thresholds', 'effect', data_all, marker='', ci=95, line_kws={'lw':lw})
plt.plot([0, 25], [0, 0], 'k', lw=lw-.25, zorder=-100)
x = np.linspace(-1, 25, 100)
plt.ylim([-6, 12])
plt.xlim([0, 25])
plt.title('All durations')
plt.xlabel(' ')
plt.ylabel(u'\nThreshold\nimprovement (ln °)')
plt.ylim([-1, 1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.text(2, 8, u'r = 0.29 \np = 1e-2', fontsize=fontsize, color='C0')
plt.tight_layout()
plt.savefig('paper4.pdf', dpi=600)

plt.figure(figsize=(3, 1.75), dpi=600)
x = np.linspace(-1, 25, 100)
data = read_hdf5('original_diff.hdf5')
cont = data['thresh']
diffs = data['diffs']
plt.plot(cont, diffs, '+', c='C0', ms=ms, label='original experiment')
data = DataFrame(data={'control_thresholds': cont.ravel(), 'effect': diffs.ravel()})
err = sns.regplot('control_thresholds', 'effect', data,ci=95, color='C0', marker='', line_kws={'lw':lw})
plt.plot([0, 25], [0, 0], 'k', lw=lw-.25, zorder=-100)
plt.xlabel(u'Central threshold (°)')
plt.ylabel(u'\nThreshold\nimprovement (°)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([-6, 12])
plt.xlim([0, 25])
x = np.linspace(-1, 25, 100)
plt.title('Previous study (300 ms only)')

plt.text(19, -5.3, 'r = 0.76 \np = 1e-4', fontsize=fontsize, color='C0')
plt.tight_layout()
#plt.subplots_adjust(left=0.082, right=0.98, top=0.877, bottom=0.215)

plt.savefig('paper5.pdf', dpi=600)

plt.savefig('summary_lin_fits.pdf', dpi=600)

# %%
plt.figure(figsize=(2, 2))
x = np.linspace(-1, 25, 100)
data = read_hdf5('original_diff.hdf5')
cont = data['thresh']
diffs = data['diffs']
plt.plot(cont, diffs, 'o', c='C0', ms=2, label='original experiment')
data = DataFrame(data={'control_thresholds': cont.ravel(), 'effect': diffs.ravel()})
err = sns.regplot('control_thresholds', 'effect', data,ci=95, color='k', marker='', line_kws={'linewidth': 1, 'color': 'C0', 'marker': ''})
plt.plot([0, 25], [0, 0], 'k')
plt.xlabel(u'Central threshold (°)')
plt.ylabel(u'Threshold improvement (°)')
#plt.ylabel('')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim([-2, 6])
plt.xlim([0, 18])
x = np.linspace(-1, 25, 100)
plt.tight_layout()
plt.savefig('previous_data.png', dpi=600)

# %%
fig = plt.figure(figsize=(.5, 1.75))

kernel = gaussian_kde(effect.ravel()[~np.isnan(effect.ravel())], 0.3)
plt.fill_betweenx(np.arange(-6, 12, .01), 20 * kernel(np.arange(-6, 12, .01)), color='C0')
#plt.plot(20 * kernel(np.arange(-6, 12, .01)), np.arange(-6, 12, .01), color='C0', zorder=100)
plt.plot([0, 10], [0, 0], 'w', lw=lw-.25, zorder=100)
plt.ylim([-6, 12])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xlabel(' ')
plt.tight_layout()
plt.subplots_adjust(left=0, right=0.98, top=0.81, bottom=0.302)
plt.savefig('breakout_marginals_new.pdf', dpi=600)
# %%
fig = plt.figure(figsize=(.5, 1.75))
kernel = gaussian_kde(diffs.ravel()[~np.isnan(diffs.ravel())], 0.3)
plt.fill_betweenx(np.arange(-6, 12, .01), 20 * kernel(np.arange(-6, 12, .01)), color='C0')
#plt.plot(20 * kernel(np.arange(-6, 12, .01)), np.arange(-6, 12, .01), color='k', zorder=100)
plt.plot([0, 10], [0, 0], 'w', lw=lw-.25, zorder=100)
plt.ylim([-6, 12])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xlabel(' ')
plt.tight_layout()
plt.subplots_adjust(left=0, right=0.98, top=0.81, bottom=0.302)
plt.savefig('breakout_marginals_old.pdf', dpi=600)
#plt.tight_layout()

#plt.figure()
#x = np.arange(-2, 5, .01)
#plt.subplot(211)
#kernel_300 = gaussian_kde(np.log(control_thresholds[:, 1]))
#kernel_orig = gaussian_kde(np.log(cont))
#plt.fill_between(x, kernel_300(x), alpha=0.5)
#
#plt.fill_between(x, kernel_orig(x), alpha=0.5)
#
#plt.xlabel('LOG central threshold')
#plt.legend(('300 ms', 'original'), loc='best')
#plt.subplot(212)
#kernel_300 = gaussian_kde(np.log(thresholds[:, 1, :].squeeze()[:, 1]))
#kernel_orig = gaussian_kde(np.log(cont - diffs.squeeze()))
#plt.fill_between(x, kernel_300(x), alpha=0.5)
#
#plt.fill_between(x, kernel_orig(x), alpha=0.5)
#
#plt.xlabel('LOG matched threshold')

# %%
#plt.figure()
#
##plt.plot(10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[0], zorder=99)
#
##plt.plot(1 + 10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[1], zorder=100)
#
##plt.plot(2 + 10 * kernel(np.arange(-6, 12, .01))[kernel(np.arange(-6, 12, .01)) > 0.005], np.arange(-6, 12, .01)[kernel(np.arange(-6, 12, .01)) > 0.005], color=colors[2], zorder=99)
#plt.plot([0, 10], [0, 0], 'w', zorder=101)
#plt.ylim([-6, 12])
#plt.gca().xaxis.set_visible(False)
#plt.gca().yaxis.set_visible(False)
#plt.gca().spines['top'].set_visible(False)
#plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
# %%
plt.figure()
for e, co, pv in zip(effect, control_thresholds, pvals_rev[:, :, 0]):
    plt.plot(co, e, 'o', c='C0', ms=ms)
    plt.plot(co, e, 'o', c='w', ms=ms-2)
    plt.plot(co[pv < 0.025], e[pv < 0.025], 'o', c='C0', ms=ms)
    plt.plot(co[pv > 0.975], e[pv > 0.975], 'o', c='C0', ms=ms)
plt.plot(cont, diffs, '+', c='k', ms=ms, label='original experiment')
data_both = DataFrame(data={'control_thresholds': np.concatenate((control_thresholds.ravel(), cont.ravel()), 0), 'effect': np.concatenate((effect.ravel(), diffs.ravel()), 0)})
err = sns.regplot('control_thresholds', 'effect', data_both, marker='', ci=95)
plt.plot([0, 25], [0, 0], 'k')
x = np.linspace(-1, 25, 100)
plt.ylim([-6, 12])
plt.xlim([0, 25])
plt.title('All durations')
plt.xlabel(u'Central threshold (°)')
plt.ylabel(u'Threshold improvement (°)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.text(1, 9, 'r = 0.56 \np = 4e-7', fontsize=fontsize, color='C0')
x_all = data_both.values[:, 0]
y_all = data_both.values[:, 1]
slope, intercept, r_value, p_value, std_err = linregress(x_all[~np.isnan(y_all)], y_all[~np.isnan(y_all)])
print(p_value, r_value)

# %%
plt.figure()
shift = ss.rankdata(thresholds[:, 0, :].mean(-1)) - len(subjects) / 2
shift *= .02 / max(shift)
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
          'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
shapes = 10 * ['o']
sh = 10 * ['s']
shapes = shapes + sh
for e, s, c, sh in zip(effect, shift, colors, shapes):
    plt.stem(durations + s, e, c, markerfmt=sh, basefmt='k')
    plt.plot(durations + s, e, sh, c=c, ms=8)
plt.plot([0, 1.1], [0, 0], 'k', zorder=-50)
plt.xlim([0, 1.1])
for m, d in zip(np.nanmean(effect, 0), durations):
    plt.plot([d - 0.025, d + 0.025], [m, m], 'k', lw=5, zorder=-50)
plt.ylabel('Change in threshold (deg)')
plt.xlabel('Duration (s)')

# %% RESIDUALS
#
#plt.figure()_
#plt.subplot(151)
#sns.residplot('control_thresholds', 'effect', data1, color='C1')
#plt.subplot(152)
#sns.residplot('control_thresholds', 'effect', data2, color='C2')
#plt.subplot(153)
#sns.residplot('control_thresholds', 'effect', data3, color='C3')
#plt.subplot(154)
#sns.residplot('control_thresholds', 'effect', data_all, color='C0')
#plt.subplot(155)
#sns.residplot('control_thresholds', 'effect', data, color='k')

# %% homoscedasticity
from statsmodels.stats.diagnostic import het_breushpagan


slope, intercept, r_value, p_value, std_err = linregress(cont, diffs.squeeze())
residual = cont * slope + intercept - diffs.squeeze()

_, pval, _, f_pval = het_breushpagan(residual, data[['control_thresholds', 'effect']])

slope, intercept, r_value, p_value, std_err = linregress(control_thresholds[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())], effect[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())])
print(p_value, r_value)
residual = control_thresholds[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())] * slope + intercept - effect[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())]
data1 = DataFrame(data={'control_thresholds': control_thresholds[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())], 'effect': effect[:, 0].ravel()[~np.isnan(effect[:, 0].ravel())]})
_, pval1, _, f_pval = het_breushpagan(residual, data1[['control_thresholds', 'effect']])

inds = np.arange(20)
inds = np.delete(inds, [1, 16])
slope, intercept, r_value, p_value, std_err = linregress(control_thresholds[:, 1].ravel()[inds], effect[:, 1].ravel()[inds])
print(p_value, r_value)
residual = control_thresholds[:, 1].ravel() * slope + intercept - effect[:, 1].ravel()
_, pval2, _, f_pval = het_breushpagan(residual, data2[['control_thresholds', 'effect']])

inds = np.arange(20)
inds = np.delete(inds, [15, 8])
slope, intercept, r_value, p_value, std_err = linregress(control_thresholds[:, 2].ravel()[inds], effect[:, 2].ravel()[inds])
print(p_value, r_value)
residual = control_thresholds[:, 2].ravel() * slope + intercept - effect[:, 2].ravel()
_, pval3, _, f_pval = het_breushpagan(residual, data3[['control_thresholds', 'effect']])


slope, intercept, r_value, p_value, std_err = linregress(control_thresholds.ravel()[~np.isnan(effect.ravel())], effect.ravel()[~np.isnan(effect.ravel())])
print(p_value, r_value)
slope, intercept, r_value, p_value, std_err = linregress(cont, diffs.T)
print(p_value, r_value)
## %% Individual subject plots
#plt.figure()
#for i, threshold in enumerate(thresholds):
#    plt.subplot(4, 5, i+1)
#
#    plt.plot([.1, .3, 1], threshold.T, '-o')
#    plt.xlim([0, 1.1])
#    plt.ylim([0, 25])
#    plt.title('Sub ' + subjects[i])
#    if i==0:
#        plt.legend(('Central', 'Matched'))
#    if i==17:
#        plt.xlabel('Duration (s)')
#    if i==5:
#        plt.ylabel(u'Threshold (°)')
##plt.figure()
##plt.plot(thresholds[:, 0, :].ravel(), 'o')
##plt.plot(thresholds[:, 1, :].ravel(), 's')
##plt.plot(-1, np.nanmean(thresholds[:, 0, :].ravel()), 'o', c='C0', ms=12)
##plt.plot(-1, np.nanmean(thresholds[:, 1, :].ravel()), 's', c='C1', ms=12)
#
## %% 
#
#plt.figure()
#plt.subplot(121)
#plt.plot([.1, .3, 1], thresholds[:, 0].T, '-o', c='C0', alpha=0.3)
#plt.plot([.1, .3, 1], np.nanmean(thresholds[:, 0], 0), 'o', ms=10, c='C0')
#plt.ylim([0, 27])
#plt.subplot(122)
#plt.plot([.1, .3, 1], thresholds[:, 1].T, '-o', c='C1', alpha=0.3)
#plt.plot([.1, .3, 1], np.nanmean(thresholds[:, 1], 0), 'o', ms=10, c='C1')
#plt.ylim([0, 27])
## %%
#plt.figure(figsize=(3,3))
#plt.plot(thresholds[:10, 0].T, thresholds[:10, 1].T, '-o')
#plt.plot(thresholds[10:, 0].T, thresholds[10:, 1].T, '--s')
#plt.plot([0, 27], [0, 27], 'k')
#plt.ylim([0, 27])
#plt.xlim([0, 27])
#plt.xlabel('Central threshold')
#plt.ylabel('Matched threshold')
#
## %% 
#plt.figure()
#data_con = DataFrame(data={'duration':np.array(40 * [.1, .3, 1]),'condition':np.array([60 * ['control'], 60 * ['matched']]).ravel(), 'threshold':thresholds.ravel()})
#sns.violinplot(x='duration', y='threshold', hue='condition', data=data_con, split=True)

# %%
#plt.figure()
#plt.subplot(121)
#min_th = []
#min_con = []
#for x, sub in enumerate(effect):
#    sub2 = sub[~np.isnan(sub)]
#    th = control_thresholds[x][~np.isnan(sub)]
#    min_con.append(th[-1])
#    min_th.append(sub2[-1])
#    plt.plot(th[-1], sub2[-1], '^g')
#plt.plot([0, 25], [0, 0], 'k')
#plt.xlim([0, 25])
#plt.ylim([-6, 12])
#plt.xlabel('Smallest central threshold')
#plt.ylabel('Effect size (deg)')
#plt.tight_layout()
#plt.subplot(122)
#max_th = []
#max_con = []
#for x, sub in enumerate(effect):
#    sub2 = sub[~np.isnan(sub)]
#    th = control_thresholds[x][~np.isnan(sub)]
#    max_th.append(sub2[0])
#    max_con.append(th[0])
#    plt.plot(th[0], sub2[0], '^g')
#plt.plot([0, 25], [0, 0], 'k')
#plt.xlim([0, 25])
#plt.ylim([-6, 12])
#plt.xlabel('Largest central threshold')
#plt.ylabel('Effect size (deg)')
#plt.tight_layout()
#
#pvals_2 = [ttest_rel(thresholds[:, 0, i], thresholds[:, 1, i], nan_policy='omit') for i in range(3)]

# %%
#plt.figure()
#for min_t, max_t, min_c, max_c in zip(min_th, max_th, min_con, max_con):
#    plt.plot([min_c, max_c], [min_t, max_t], '-o')
## %%
#plt.figure()
#plt.plot([np.mean([ma, mi]) for ma, mi in zip(max_con, min_con)], [ma - mi for ma, mi in zip(max_th, min_th)], 'o')
## %%
#plt.figure()
#plt.plot([.1, .3, 1], effect.T, '-o')
#
## %%
#data = DataFrame(data={'duration':np.array(20 * [.1, .3, 1]),'effect':effect.ravel()})
#sns.boxplot(x='duration', y='effect', data=data, notch=True)
#plt.plot([-1, 3], [0, 0], 'k')
## %%
#sns.swarmplot('duration', 'effect', data=data, size=5)
#sns.pointplot('duration', 'effect', data=data, join=False, markers ='^', scale=.1, capsize=.6, errwidth=1)
#
## %%
#plt.figure()
##plt.subplot(211)
all_subs = np.array([0, 2, 5, 6, 9, 10, 11, 12, 13, 17, 18])
some_subs = np.array([1, 3, 4, 7, 8, 14, 15, 16, 19])
#plt.plot([.1, .3, 1], effect[all_subs].T, '-ok')
#plt.subplot(212)
#plt.plot([.1, .3, 1], effect[np.isnan(effect[:, 0])].T)

# %%
durs = [100, 300, 1000]
plt.figure(figsize=(3.5, 3.5))
th_3 = thresholds[all_subs]
for t in th_3:
    plt.subplot(221)
    plt.plot(durs, t[0], '-', c=3 * [.5], ms=5, lw=.5)
    plt.subplot(223)
    plt.plot(durs, t[1], '-', c=3 * [.5], ms=5, lw=.5)

plt.subplot(221)

plt.plot(durs, th_3[:, 0].mean(0),'d-k', ms=3, lw=1)
plt.errorbar(durs, th_3[:, 0].mean(0), th_3[:, 0].std(0)/np.sqrt(11), c='k', ms=1, lw=1)
plt.ylabel(u'Central Threshold (°)')
plt.xticks(durs)
#plt.xlabel('Duration (ms)')
plt.title('Subjects with\nall thresholds')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim([0, 1100])
plt.ylim([0, 27])
plt.subplot(223)
plt.xlim([0, 1.1])
plt.ylim([0, 27])
plt.plot(durs, th_3[:, 0].mean(0),'d-k', ms=3, lw=1)
plt.errorbar(durs, th_3[:, 0].mean(0), th_3[:, 0].std(0)/np.sqrt(11), c='k', ms=1, lw=1)
plt.xticks(durs)
plt.xlabel('Duration (ms)')
plt.ylabel(u'Matched Threshold (°)')
#plt.title('Matched')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

th_3 = thresholds[some_subs]
plt.xlim([0, 1100])
for t in th_3:
    plt.subplot(222)
    plt.plot(durs, t[0], '-', c=3 * [.5], ms=5, lw=.5)
    plt.subplot(224)
    plt.plot(durs, t[1], '-', c=3 * [.5], ms=5, lw=.5)

plt.subplot(222)
plt.xlim([0, 1.1])
plt.ylim([0, 27])
plt.plot(durs, th_3[:, 0].mean(0),'d-k', ms=3, lw=1)
plt.errorbar(durs, th_3[:, 0].mean(0), th_3[:, 0].std(0)/np.sqrt(11), c='k', ms=1, lw=1)
plt.xlim([0, 1100])
plt.xticks(durs)
#plt.xlabel('Duration (ms)')
plt.title('Subjects with\nmissing thresholds')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.subplot(224)
plt.xlim([0, 1.1])
plt.ylim([0, 27])
plt.plot(durs, th_3[:, 0].mean(0),'d-k', ms=3, lw=1)
plt.errorbar(durs, th_3[:, 0].mean(0), th_3[:, 0].std(0)/np.sqrt(11), c='k', ms=1, lw=1)
plt.xticks(durs)
plt.xlim([0, 1100])
plt.xlabel('Duration (ms)')
#plt.title('Matched')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('thresholds_saturated.pdf', dpi=600)

# %%
#plt.figure()
#for i, a in enumerate(all_subs):
#    plt.subplot(3, 4, i + 1)
#    plt.plot([.1, .3, 1], thresholds[a, 0].T, '-o')
#    plt.plot([.1, .3, 1], thresholds[a, 1].T, '--s')
#plt.tight_layout()
## %%
#plt.figure()
#improvement_dur = thresholds[all_subs, :, 0] - thresholds[all_subs, :, 2]
#
#plt.plot([0, 1], improvement_dur.T, 'o-')
#plt.plot([0, 1], improvement_dur.mean(0), 'ok')
#
## %%
#plt.figure()
#plt.plot([.1, .3, 1], thresholds[17, 0, :], '-ok')
#plt.plot([.1, .3, 1], thresholds[17, 1, :], '--s', c='C4')
#plt.fill_between([.1, .3, 1], thresholds[17, 0, :], thresholds[17, 1, :], color='C4', alpha=0.3)
#plt.gca().invert_yaxis()
#plt.legend(('Control', 'Matched', 'Visual Effect'))
#plt.ylim([0, 24])
#plt.xlabel('Duration (s)')
#plt.ylabel(u'Separation (°)')
#plt.gca().spines['top'].set_visible(False)
#plt.gca().spines['right'].set_visible(False)
#
## %%
#plt.figure()
#p = pvals_rev[17, :, 0] < 0.025
#plt.plot([.1, .3, 1], thresholds[17, 0, :], '-ok')
#plt.plot([.1, .3, 1], thresholds[17, 1, :], '--sk', alpha=0.5)
#plt.gca().invert_yaxis()
#plt.plot([np.array([.1, .3, 1])[p], np.array([.1, .3, 1])[p]], [thresholds[17, 0, :][p], thresholds[17, 1, :][p]], c='C9', alpha=0.3, lw=5)
#plt.legend(('Control', 'Matched', 'Visual Effect'))
## %%
#plt.figure()
#plt.plot([.1, .1], thresholds[17, :, 0], '-', c='C4', lw=5)
##plt.plot([.3, .3], thresholds[17, :, 1], '-k', alpha=0.5, lw=5)
#plt.plot([1, 1], thresholds[17, :, 2], '-', c='C4', lw=5)
#plt.plot([.1, .3, 1], thresholds[17, 0, :], 'ok', ms=5)
#plt.plot([.1, .3, 1], thresholds[17, 1, :], 'sk', ms=5)
#
## %%
#plt.figure()
#angles = np.array([.1, .3, 1])
#th_sort = thresholds.copy()
#th_sort[np.isnan(th_sort)] = 35
#for i in range(20):
#    sub_i = np.where([np.nanmean(th_sort[i, 0, :]) == np.flipud(np.sort(np.nanmean(th_sort[:, 0, :], -1)))])[1]
#    plt.subplot(4, 5, sub_i+1)
#    plt.plot(angles[~np.isnan(thresholds[i, 1, :])], thresholds[i, 0, :][~np.isnan(thresholds[i, 1, :])], '-ok')
#    plt.plot(angles[~np.isnan(thresholds[i, 0, :])], thresholds[i, 1, :][~np.isnan(thresholds[i, 0, :])], '--k', alpha=0.5)
#    p = [p_r < 0.025 or p_r > 0.975 for p_r in pvals_rev[i, :, 0]]
#    for e, sig, t, a in zip(effect[i], p, thresholds[i, 1, :], angles):
#        if e > 0:
#            plt.plot(a, t, '^', mec='C0', mfc='w')
#            if sig:
#                plt.plot(a, t, '^', c='C0')
#        elif e < 0:
#            plt.plot(a, t, 'v', mec='C3', mfc='w')        
#            if sig:
#                plt.plot(a, t, 'v', c='C3')
#            
#    if sub_i==17:
#        plt.xlabel('Duration (s)')
#    if sub_i==5:
#        plt.ylabel(u'Threshold (°)')
#    plt.xlim([0, 1.1])
#    plt.ylim([0, 27])
#    plt.gca().invert_yaxis()
#plt.tight_layout()
#
# %%
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(cycle[0])
plt.figure(figsize=(3.5, 2.5))
plt.subplot(121)
plt.title('Subjects with\nall thresholds')
plt.plot(durs, effect[all_subs, :].T, '-', c=[.459, .722, .906],  ms=2, lw=.5)
#plt.plot(effect[:, 1:].mean(0), 'o-k', ms=10)
plt.errorbar(durs, effect[all_subs, :].mean(0), effect[all_subs, :].std(0) / np.sqrt(len(all_subs)), c='C0', fmt='o-', zorder=100, lw=1, ms=3)
#plt.errorbar([.3, 1], np.delete(effect[:, 1:], 15, 0).mean(0), np.delete(effect[:, 1:], 15, 0).std(0) / np.sqrt(20), fmt='o-r', zorder=100)
plt.plot([0, 2500], [0, 0], 'k', lw=lw-.25, zorder=-100)
plt.xticks(durs)
plt.xlim([0, 1100])
plt.ylim([-5, 11])
plt.yticks([-5, 0, 5, 10])
plt.xlabel('Duration (ms)')
plt.ylabel(u'Threshold improvement (°)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.subplot(122)
plt.title('Subjects with\nmissing thresholds')
some_subs = np.array([1, 3, 4, 7, 8, 14, 15, 16, 19])
plt.plot(durs, effect[some_subs, :].T, '-', c=[.459, .722, .906],  ms=2, lw=.5)
#plt.plot(effect[:, 1:].mean(0), 'o-k', ms=10)
plt.errorbar(durs[1:], effect[some_subs, 1:].mean(0), effect[some_subs, 1:].std(0) / np.sqrt(len(some_subs)), c='C0', fmt='o-', zorder=100, lw=1, ms=3)
#plt.errorbar([.3, 1], np.delete(effect[:, 1:], 15, 0).mean(0), np.delete(effect[:, 1:], 15, 0).std(0) / np.sqrt(20), fmt='o-r', zorder=100)
plt.plot([0, 2500], [0, 0], 'k', lw=lw-.25, zorder=-100)
plt.xticks(durs)
plt.xlim([0, 1100])
plt.ylim([-5, 11])
plt.xlabel('Duration (ms)')
#plt.ylabel(u'Threshold improvement (°)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([-5, 0, 5, 10])
plt.tight_layout()
plt.savefig('compare_durations.pdf', dpi=600)
#
stop

# %%

import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

subs = np.array([3 * [i] for i in np.arange(20)], dtype=int).ravel()
durs = np.array(20 * [100, 300, 1000], dtype=int)
mod_eff = effect.ravel()

data_array = dict(Subject=subs,
                  Duration=durs,
                  Effect=mod_eff)
formula = "Effect ~ C(Duration) + C(Subject)"
formula = "Effect ~ Duration"
df = pd.DataFrame(data_array)

model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2, robust='hc3')
print(aov_table)


data_array = dict(Subject=subs[~np.isnan(mod_eff)],
                  Duration=durs[~np.isnan(mod_eff)],
                  Effect=mod_eff[~np.isnan(mod_eff)])
df = pd.DataFrame(data_array)
md = smf.mixedlm(formula, df, groups=df["Subject"])
mdf = md.fit()
print(mdf.summary())

# %%

subs = np.array([6 * [i] for i in np.arange(20)], dtype=int).ravel()
durs = np.array(40 * [100, 300, 1000], dtype=int)
condition = np.array(30 * [0, 0, 1, 1], dtype=int).ravel()
thresholds = np.log(thresholds).ravel()

data_array = dict(Subject=subs,
                  Vis=condition,
                  Duration=durs,
                  Effect=thresholds)
formula = "Effect ~ C(Duration) + C(Subject)"
formula = "Effect ~ Duration + C(Vis) + Duration:C(Vis)"
df = pd.DataFrame(data_array)

model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2, robust='hc3')
print(aov_table)

data_array = dict(Subject=subs[~np.isnan(thresholds)],
                  Vis=condition[~np.isnan(thresholds)],
                  Duration=durs[~np.isnan(thresholds)],
                  Effect=thresholds[~np.isnan(thresholds)])
df = pd.DataFrame(data_array)
md = smf.mixedlm(formula, df, groups=df["Subject"])
mdf = md.fit()
print(mdf.summary())

from statsmodels.graphics.factorplots import interaction_plot
interaction_plot(df.Duration, df.Vis, df.Effect)
#ttest_rel(np.delete(effect[:, 1], 15), np.delete(effect[:, 2], 15))
#ttest_1samp(np.delete(effect[:, 1], 15), 0)
#ttest_1samp(np.delete(effect[:, 2], 15), 0)
#ttest_rel(effect[:, 1], effect[:, 2])
#ttest_1samp(effect[:, 1], 0)
#ttest_1samp(effect[:, 2], 0)
## %%
#shift = ss.rankdata(thresholds[:, 0, :].mean(-1)) - 20 / 2
#shift *= .02 / max(shift)
#colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#          'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
#shapes = 10 * ['-o']
#sh = 10 * ['-s']
#shapes = shapes + sh
#for e, s, c, sh in zip(effect, shift, colors, shapes):
#    #    plt.stem(durations + s, e, c, fmt=sh, basefmt='k')
#    plt.plot(durations + s, e, sh, c=c, ms=8)
#plt.plot([0, 1.1], [0, 0], 'k', zorder=-50)
#plt.xlim([0, 1.1])
#for m, d in zip(np.nanmean(effect, 0), durations):
#    plt.plot([d - 0.025, d + 0.025], [m, m], 'k', lw=5, zorder=-50)
#plt.plot([0, 1.1], [0, 0], 'k')
#plt.xlim([0, 1.1])
#plt.ylabel('Change in threshold (deg)')
#plt.xlabel('Duration (s)')
#
## %%
#plt.figure()
#plt.plot(np.nanmean(control_thresholds, -1), np.nanmean(effect, -1), 'o')
#plt.xlabel('Mean cental threshold')
#plt.ylabel('Mean threshold improvement')
#slope, intercept, r_value, p_value, std_err = linregress(np.nanmean(control_thresholds, -1), np.nanmean(effect, -1))
#print(p_value, r_value)
