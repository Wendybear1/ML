from __future__ import division
import mne
import numpy as np
import pycwt
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter, iirfilter, filtfilt
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from matplotlib import rc
import csv
import datetime
import pytz
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import scipy
from datetime import datetime, timedelta
from pycwt import cwt


def butter_bandpass(lowcut, highcut, fs,order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs,order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def Implement_Notch_Filter(fs, band, freq, ripple, order, filter_type, data):
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def sliding_window(elements, window_size,overlap):
    # if len(elements) <= window_size:
    #     # segment_arr.append(elements)
    #     return elements
    segment_arr = []
    for i in range(0,len(elements),overlap):
        # print(elements[i:i + window_size])
        segment_arr.append(elements[i:i + window_size])
    return segment_arr

# sigal=[3,2,5,67,1,23,56,5,34,22,1,2,4,5]
# a=sliding_window(sigal,3,3)
# print(a)





# Fs = 256
# f = 1
# sample = 256*10
# x1 = np.arange(sample)
# y1 = np.sin(2 * np.pi * f * x1 / Fs)
# Fs = 256
# f = 1.5
# sample = 256*10
# x2 = np.arange(sample)
# y2 = np.sin(2 * np.pi * f * x2 / Fs)
# y=y1+y2
# pyplot.plot(x1, y,'k',label='delta')
# Fs = 256
# f = 15
# sample = 256*10
# x3 = np.arange(sample)
# y3 = np.sin(2 * np.pi * f * x3 / Fs)
# y=y1+y2+y3
# pyplot.plot(x1, y,'grey',label='noised')
# sig_filtered = butter_bandpass_filter(y, 0.5, 2, 256, order=4)
# pyplot.plot(x1, sig_filtered,'r', label='filtered')
# pyplot.xlabel('sample(n)')
# pyplot.ylabel('voltage(V)')
# pyplot.legend()
# pyplot.show()


import os
channel = ['Fz', 'C4', 'Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'A2', 'T4', 'A1', 'T3', 'Fp1', 'Fp2', 'O2', 'O1', 'F7', 'F8',
           'T6', 'T5', 'Cz']
# for i in range(len(channel)):
for i in range(1,2):
    channel_arr = []
    directory = r'C:/Users/wxiong/PycharmProjects/ML/delta'
    dir_list = list(os.scandir(directory))
    dir_list.sort(key=lambda d: d.path)
    for entry in dir_list:
        if (entry.path.endswith(".csv")) and entry.is_file():
            raw_ecg = pd.read_csv(entry.path, skipinitialspace=True)
            ch = channel[i]
            target_signal = raw_ecg[ch].values*(10**6)
            epoch_time = raw_ecg['time'].values

            df = pd.DataFrame(list(zip(epoch_time, target_signal)), columns=['time', 'wave'])
            df['time'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
            # pyplot.plot(df['time'].values, df['wave'].values,label='EEG',c='grey')
            # pyplot.ylabel('$\mathregular{\u03BCV}$',fontsize=10)
            # pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
            # pyplot.legend()
            # pyplot.show()

            segment=sliding_window(target_signal, 256*60*10,256*60*5)
            time_0=epoch_time[0]

            signal_arr=[]
            time_arr=[]
            for m in range(len(segment)):
            # for m in range(13,14):
                sig_filtered = butter_bandpass_filter(segment[m], 0.5, 2, 256, order=4)

                time=np.linspace(int(time_0/1000), int(time_0/1000) + 10*60, len(sig_filtered))
                print(len(time))

                df = pd.DataFrame(list(zip(time, sig_filtered)), columns=['time', 'wave'])
                df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
                print(time[0]);print(time[-1]);
                # pyplot.plot(df['time'].values, df['wave'].values)
                # pyplot.legend([f'{ch}'])
                # pyplot.gca().xaxis.set_major_formatter(
                #     mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #  #pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                # pyplot.show()

                if m==23:
                    pyplot.plot(df['time'].values, df['wave'].values)
                    pyplot.legend([f'{ch}'])
                    pyplot.gca().xaxis.set_major_formatter(
                        mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                    # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                    pyplot.show()

                # if m==13:
                #     pyplot.plot(df['time'].values[16600:26600], df['wave'].values[16600:26600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[26600:36600], df['wave'].values[26600:36600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[36600:46600], df['wave'].values[36600:46600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[46600:56600], df['wave'].values[46600:56600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     pyplot.ylim([-60, 75])
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.show()
                #
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[56600:66600], df['wave'].values[56600:66600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     pyplot.ylim([-60, 75])
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.show()
                # if m==13:
                #     pyplot.plot(df['time'].values[66600:76600], df['wave'].values[66600:76600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[76600:80600], df['wave'].values[76600:80600])
                #     #pyplot.plot(df['time'].values[175600:], df['wave'].values[175600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[80600:90600], df['wave'].values[80600:90600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                # if m==13:
                #     pyplot.plot(df['time'].values[90600:100600], df['wave'].values[90600:100600])
                #     #pyplot.plot(df['time'].values[175600:], df['wave'].values[175600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[100600:110600], df['wave'].values[100600:110600])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[110600:120600], df['wave'].values[110600:120600])
                #     #pyplot.plot(df['time'].values[175600:], df['wave'].values[175600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[120600:130600], df['wave'].values[120600:130600])
                #     #pyplot.plot(df['time'].values[135600:], df['wave'].values[135600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[130600:140600], df['wave'].values[130600:140600])
                #     #pyplot.plot(df['time'].values[135600:], df['wave'].values[135600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[140600:150600], df['wave'].values[140600:150600])
                #     #pyplot.plot(df['time'].values[135600:], df['wave'].values[135600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     pyplot.show()
                #
                # if m==13:
                #     pyplot.plot(df['time'].values[150600:160600], df['wave'].values[150600:160600])
                #     #pyplot.plot(df['time'].values[135600:], df['wave'].values[135600:])
                #     pyplot.legend([f'{ch}'])
                #     pyplot.gca().xaxis.set_major_formatter(
                #         mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
                #     # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                #     pyplot.ylim([-60, 75])
                #     # pyplot.show()



                time_0 = time_0 + 5 * 60000
                print(time_0)



                # signal_arr = signal_arr + list(sig_filtered)
                # time_arr = time_arr + list(time)
                # df = pd.DataFrame(list(zip(time_arr,signal_arr)), columns=['time','wave'])
                # df['time']=pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
                # pyplot.plot(df['time'].values,df['wave'].values)
                # pyplot.legend([f'{ch}'])
                # pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=pytz.timezone('Australia/Sydney')))
                # pyplot.show()



# ### wavelet
import os
channel = ['Fz', 'C4', 'Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'A2', 'T4', 'A1', 'T3', 'Fp1', 'Fp2', 'O2', 'O1', 'F7', 'F8',
           'T6', 'T5', 'Cz']
# for i in range(len(channel)):
for n in range(1,2):
    channel_arr = []
    directory = r'C:/Users/wxiong/PycharmProjects/ML/delta'
    dir_list = list(os.scandir(directory))
    dir_list.sort(key=lambda d: d.path)
    for entry in dir_list:
        if (entry.path.endswith(".csv")) and entry.is_file():
            print(entry)
            raw_ecg = pd.read_csv(entry.path, skipinitialspace=True)
            ch = channel[n]
            target_signal = raw_ecg[ch].values*(10**6)
            epoch_time = raw_ecg['time'].values

            df = pd.DataFrame(list(zip(epoch_time, target_signal)), columns=['time', 'wave'])
            df['time'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
            # pyplot.plot(df['time'].values, df['wave'].values,label='EEG',c='grey')
            # pyplot.ylabel('$\mathregular{\u03BCV}$',fontsize=10)
            # pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
            # pyplot.legend()
            # pyplot.show()

            segment=sliding_window(target_signal, 256*60*10,256*60*5)
            time_0=epoch_time[0]

            signal_arr=[]
            time_arr=[]
            # for m in range(len(segment)):
            for m in range(23,24):

                y= butter_bandpass_filter(segment[m], 0.5, 2, 256, order=4)

                 #y=segment[m]
                # y=target_signal

                # wavelet transform
                # freqs = np.append(np.arange(0.5, 2, 1), np.arange(2, 4, 1))
                # freqs = np.append(freqs, np.arange(4, 8, 1))
                # freqs = np.append(freqs, np.arange(8, 12, 1))
                # freqs = np.append(freqs, np.arange(13, 30, 1))

                freqs = np.append(np.arange(0.5, 2, 0.2), np.arange(2, 4, 0.2))

                # # print(freqs)
                # freqs = (1 / freqs)
                #
                dt = 1 / 256  # (frequency in hz)
                alpha, _, _ = pycwt.ar1(y)  # lag 1 autocorrelation for significance
                # # mother = pycwt.Morlet(6)
                mother = pycwt.DOG(6)
                # mother = pycwt.MexicanHat()

                wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(signal=y, dt=dt, wavelet=mother, freqs=freqs)
                power = np.abs(wave) ** 2
                fft_power = np.abs(fft) ** 2
                # period = 1 / freqs
                glbl_power = power.mean(axis=1)
                dof = y.size - scales  # Correction for padding at edges
                var = y.std() ** 2
                glbl_signif, fft_theor = pycwt.significance(var, dt, scales, 1, alpha, significance_level=0.99, dof=dof,
                                                            wavelet=mother)
                # Find peaks that are significant
                xpeaks = [];
                powers = []
                ind_peaks = scipy.signal.find_peaks(var * glbl_power)[0]
                for i in ind_peaks:
                    peak = [var * glbl_power > glbl_signif][0][i]
                    if peak:
                        if freqs[i] not in xpeaks:
                            xpeaks.append(freqs[i])
                            powers.append([var * glbl_power][0][i])
                print(peak);
                print(powers);
                print(xpeaks);
                # print(len(freqs));

                # plt.plot(period, var * glbl_power)
                # plt.plot(period, glbl_signif, 'blue')
                # plt.show()

                # fig, ax = pyplot.subplots()
                # pyplot.plot(freqs, var * glbl_power)
                # # pyplot.plot(period, glbl_signif, 'blue')
                # # pyplot.xlabel('hours',fontsize=13)
                # pyplot.ylabel('wavelet power', fontsize=13)
                # # pyplot.title('Heart rate cycles',fontsize=12)
                # # locs, labels = pyplot.xticks(fontsize=11)
                # locs, labels = pyplot.yticks(fontsize=11)
                # pyplot.scatter(xpeaks, powers)
                # ax.spines["right"].set_visible(False)
                # ax.spines["top"].set_visible(False)
                # for i, txt in enumerate(xpeaks):
                #     pyplot.annotate(int(txt) + 1, (xpeaks[i], powers[i]), fontsize=12)
                # pyplot.show()

                N = len(y)
                sig99 = np.ones([1, N]) * glbl_signif[:, None]
                sig99 = power / sig99
                t = np.arange(0, N) * dt
                ax = pyplot.axes([0.1, 0.75, 0.85, 0.15])
                dj = 1 / 12
                iwave = pycwt.icwt(wave, scales, dt, dj, mother) * y.std()
                # ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
                ax.plot(t, y, 'grey', linewidth=1.5)
                ax.set_xlabel(r'Time(s)', fontsize=10)


                bx = pyplot.axes([0.1, 0.17, 0.85, 0.4], sharex=ax)
                # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16,32]
                levels = [128,256,256*2, 256*4, 256*8,256*16,256*32,256*64, 256*128, 256*256]
                # bx.contourf(t, np.log2(freqs), np.log2(power), np.log2(levels),
                #             extend='both', cmap=pyplot.cm.viridis)
                bx.contourf(t, freqs, np.log2(power), np.log2(levels),
                            extend='both', cmap=pyplot.cm.viridis)
                extent = [t.min(), t.max(), 0, max(freqs)]
                # bx.contour(t, np.log2(freqs), sig99, [-9, 1], colors='k', linewidths=2,
                #            extent=extent)
                # bx.contour(t, freqs, sig99, [-9, 1], colors='k', linewidths=2,
                #            extent=extent)
                # bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                #                         t[:1] - dt, t[:1] - dt]),
                #         np.concatenate([np.log2(coi), [1e-9], np.log2(freqs[-1:]),
                #                         np.log2(freqs[-1:]), [1e-9]]),
                #         'k', alpha=0.3, hatch='x')
                bx.set_ylabel('Frequency(Hz)', fontsize=10)
                # Yticks = 2 ** np.arange(np.ceil(np.log2(freqs.min())),
                #                         np.ceil(np.log2(freqs.max())))
                # bx.set_yticks(np.log2(Yticks))
                # bx.set_ylim(np.log2([freqs.min(), freqs.max()]))
                # bx.set_yticklabels(Yticks)

                # Yticks =  np.arange(0,np.ceil(freqs.max()),4)
                Yticks = np.arange(0, np.ceil(freqs.max()), 1)
                bx.set_yticks(Yticks)
                bx.set_ylim([freqs.min(), freqs.max()])
                bx.set_yticklabels(Yticks)

                # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
                # noise spectra. Note that period scale is logarithmic.
                # cx = pyplot.axes([0.77, 0.17, 0.2, 0.4], sharey=bx)
                # cx.plot(glbl_signif, np.log2(period), 'k--')
                # cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
                # cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
                #         linewidth=1.)
                # cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
                # cx.set_title('Global Wavelet Spectrum',fontsize=8)
                # cx.set_xlabel(r'Power')
                # cx.set_xlim([0, glbl_power.max() + var])
                # cx.set_ylim(np.log2([period.min(), period.max()]))
                # cx.set_yticks(np.log2(Yticks))
                # cx.set_yticklabels(Yticks)
                # pyplot.setp(cx.get_yticklabels(), visible=False)
                # for i, txt in enumerate(xpeaks):
                #     pyplot.annotate(int(txt)+1, (powers[i], np.log2(xpeaks[i])-0.3), color='r',fontsize=8)

                # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
                pyplot.show()
                # print(powers)
                # print(np.log2(xpeaks))





























# import os
# channel = ['Fz', 'C4', 'Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'A2', 'T4', 'A1', 'T3', 'Fp1', 'Fp2', 'O2', 'O1', 'F7', 'F8',
#            'T6', 'T5', 'Cz']
# #for i in range(len(channel)):
# for i in range(1,2):
#     channel_arr = []
#     directory = r'C:/Users/wxiong/PycharmProjects/ML/delta'
#     dir_list = list(os.scandir(directory))
#     dir_list.sort(key=lambda d: d.path)
#     for entry in dir_list:
#         if (entry.path.endswith(".csv")) and entry.is_file():
#             raw_ecg = pd.read_csv(entry.path, skipinitialspace=True)
#             ch = channel[i]
#             target_signal = raw_ecg[ch].values*(10**6)
#             epoch_time = raw_ecg['time'].values
#             segment=sliding_window(target_signal, 256*60*5,256*60*2)
#             time_0=epoch_time[0]
#
#             signal_arr=[]
#             time_arr=[]
#             for m in range(len(segment)):
#             #for m in range(1):
#                 sig_filtered = butter_bandpass_filter(segment[m], 0.5, 2, 256, order=4)
#
#                 time=np.linspace(int(time_0/1000), int(time_0/1000) + 5*60, len(sig_filtered))
#
#
#                 df = pd.DataFrame(list(zip(time, sig_filtered)), columns=['time', 'wave'])
#                 df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
#                 print(time[0]);print(time[-1]);
#                 pyplot.plot(df['time'].values, df['wave'].values)
#                 pyplot.legend([f'{ch}'])
#                 pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
#                 #pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
#                 pyplot.show()
#
#                 signal_arr = signal_arr + list(sig_filtered)
#                 time_arr = time_arr + list(time)
#
#                 time_0 = time_0 + 2*60000
#                 print(time_0)
#                 df = pd.DataFrame(list(zip(time_arr,signal_arr)), columns=['time','wave'])
#                 df['time']=pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
#
#                 # pyplot.plot(df['time'].values,df['wave'].values)
#                 # pyplot.legend([f'{ch}'])
#                 # pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=pytz.timezone('Australia/Sydney')))
#                 # pyplot.show()




# import os
# channel = ['Fz', 'C4', 'Pz', 'C3', 'F3', 'F4', 'P4', 'P3', 'A2', 'T4', 'A1', 'T3', 'Fp1', 'Fp2', 'O2', 'O1', 'F7', 'F8',
#            'T6', 'T5', 'Cz']
# mylist=[0,1,5,17,9,20,8,13]
# fig, axs = pyplot.subplots(2, 2)
# j=0
# #for i in range(4):
# for i in mylist:
#     channel_arr = []
#     directory = r'C:/Users/wxiong/PycharmProjects/ML/delta'
#     dir_list = list(os.scandir(directory))
#     dir_list.sort(key=lambda d: d.path)
#     for entry in dir_list:
#         if (entry.path.endswith(".csv")) and entry.is_file():
#             raw_ecg = pd.read_csv(entry.path, skipinitialspace=True)
#             ch = channel[i]
#             target_signal = raw_ecg[ch].values*(10**6)
#             epoch_time = raw_ecg['time'].values
#             segment=sliding_window(target_signal, 256*60*10,256*60*5)
#             time_0=epoch_time[0]
#             print(len(segment))
#             signal_arr=[]
#             time_arr=[]
#             # for m in range(len(segment)):
#             for m in range(27,28):
#                 sig_filtered = butter_bandpass_filter(segment[m], 0.5, 2, 256, order=4)
#
#                 time=np.linspace(int(time_0/1000), int(time_0/1000) + 10*60, len(sig_filtered))
#
#
#                 df = pd.DataFrame(list(zip(time, sig_filtered)), columns=['time', 'wave'])
#                 df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
#                 axs = axs.flatten()
#                 axs[j].plot(df['time'].values, df['wave'].values)
#                 axs[j].legend([f'{ch}'])
#                 axs[j].spines['right'].set_visible(False)
#                 axs[j].spines['top'].set_visible(False)
#                 axs[j].set_xticks([])
#                 #axs[j].set_ylim([-15, 15])
#                 #axs[j].set_yticks([])
#
#                 j=j+1
#
#                 # pyplot.plot(df['time'].values, df['wave'].values)
#                 # pyplot.legend([f'{ch}'])
#                 # pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Australia/Sydney')))
#                 # # pyplot.savefig(f'C:/Users/wxiong/PycharmProjects/ML/delta/figure/{m}.png')
#                 # pyplot.show()
#
#                 signal_arr = signal_arr + list(sig_filtered)
#                 time_arr = time_arr + list(time)
#
#                 time_0 = time_0 + 5*60000
#
#                 df = pd.DataFrame(list(zip(time_arr,signal_arr)), columns=['time','wave'])
#                 df['time']=pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Australia/Sydney')
#
#
# pyplot.show()

