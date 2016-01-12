#!/usr/bin/env python

import argparse
import copy
import ephem
import fileinput
import math
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import requests
import sys
# import time

from datetime import datetime
from datetime import timedelta
from matplotlib.pyplot import figure
from matplotlib.pyplot import grid
from matplotlib.pyplot import rc
from matplotlib.pyplot import rcParams
from mpl_toolkits.basemap import Basemap

__author__ = 'amuls'


# exit codes
E_SUCCESS = 0
E_FILE_NOT_EXIST = 1
E_NOT_IN_PATH = 2
E_UNKNOWN_OPTION = 3
E_TIME_PASSED = 4
E_WRONG_OPTION = 5
E_SIGNALTYPE_MISMATCH = 6
E_DIR_NOT_EXIST = 7
E_TIMING_ERROR = 8
E_REQUEST_ERROR = 9
E_FAILURE = 99


class Station(ephem.Observer):
    def __init__(self):
        self.name = ''
        super(Station, self).__init__()
        # ephem.Observer.__init__(self)

    def init(self, name, lat, lon, date):
        self.name = name
        self.lat = ephem.degrees(lat)
        self.lon = ephem.degrees(lon)
        self.date = ephem.date(date)

    def parse(self, text):
        elems = filter(None, re.split(',', text))
        if np.size(elems) is 3:
            self.name = elems[0]
            self.lat = ephem.degrees(elems[1])
            self.lon = ephem.degrees(elems[2])
            # self.date = ephem.date(elems[3])
        else:
            sys.stderr.write('wrong number of elements to parse\n')

    def getYMD(self):
        dateTxt = ephem.date(self.date).triple()
        year = int(dateTxt[0])
        month = int(dateTxt[1])
        day = int(dateTxt[2])
        return year, month, day

    def statPrint(self):
        yr, mm, dd = self.getYMD()
        print('%s,%s,%s,%04d/%02d/%02d' % (self.name, ephem.degrees(self.lat), ephem.degrees(self.lon), yr, mm, dd))


def loadTLE(TLEFileName, verbose=False):
    """
    Loads a TLE file and creates a list of satellites.
    Parameters:
        TLEFileName: name of TLE file
    Returns:
        listSats: List from satellites decoded from TLE
    """
    f = open(TLEFileName)
    listSats = []
    l1 = f.readline()
    while l1:
        l2 = f.readline()
        l3 = f.readline()
        sat = ephem.readtle(l1, l2, l3)
        listSats.append(sat)
        if verbose:
            print('  decoded TLE for %s' % sat.name)
        l1 = f.readline()

    f.close()
    if verbose:
        print("  %i satellites loaded into list\n" % len(listSats))

    return listSats


def setObserverData(station, predictionDate, verbose):
    '''
    setObserverData sets the info for the station from which the info is calculated
    Parameters:
        station: info over the observation station (name, lat, lon)
        predictionDate: date for doing the prediction
    Returns:
        observer contains all info about location and date for prediction
    '''
    # read in the station info (name, latitude, longitude) in degrees
    observer = Station()
    if station is None:
        observer = RMA
    else:
        observer.parse(station)

    # read in the predDate
    if predictionDate is None:
        observer.date = ephem.date(ephem.now())  # today at midnight for default start
    else:
        observer.date = ephem.Date(predictionDate)
    # print('observer.date: %04d/%02d/%02d\n' % ephem.date(observer.date).triple())

    if verbose:
        observer.statPrint()

    return observer


def setObservationTimes(observer, timeStart, timeEnd, intervalMin, verbose=False):
    '''
    observationTimes calculates the times for which the predictions will be calculated
    Parameters:
        observer has info about the date
    Returns:
        obsDates: the list of prediction times
    '''
    yyyy, mm, dd = observer.getYMD()
    # print('timeStart = %s' % (timeStart.split(':')))
    startHour, startMin = map(int, timeStart.split(':'))
    endHour, endMin = map(int, timeEnd.split(':'))
    startDateTime = datetime(yyyy, mm, dd, hour=startHour, minute=startMin, second=0, microsecond=0, tzinfo=None)
    endDateTime = datetime(yyyy, mm, dd, hour=endHour, minute=endMin, second=0, microsecond=0, tzinfo=None)
    if endDateTime <= startDateTime:
        sys.stderr.write('end time %s is less than start time %s. Program exits.\n' % (endDateTime, startDateTime))
        sys.exit(E_TIMING_ERROR)

    dtDateTime = endDateTime - startDateTime
    # print('dtDateTime = %s' % dtDateTime)
    dtMinutes = dtDateTime.total_seconds() / 60  # / timedelta(minutes=1)
    # print('dtMinutes = %s' % dtMinutes)
    nrPredictions = int(dtMinutes / float(intervalMin)) + 1
    obsDates = [startDateTime + timedelta(minutes=(int(intervalMin) * x)) for x in range(0, nrPredictions, 1)]

    if verbose:
        print('Observation time span from %s to %s with interval %d min (#%d)' % (obsDates[0], obsDates[-1], intervalMin, np.size(obsDates)))

    return obsDates, nrPredictions


def getTLEfromNORAD(TLEBaseName, verbose=False):
    '''
    getTLEfromNORAD checks whether we have a Internet connection, if yes, download latest TLE for satellite system, else check to reuse
    Parameters:
        TLEBaseName is basename of TLE file (cfr NORAD site)
    Returns:
        outFileName: filename of downloaded/reused TLE file
    '''
    if verbose:
        print('Downloading TLEs from NORAD for satellite systems %s' % TLEBaseName)

    # determine whether a list of constellations is given, if so, split up
    satSystems = TLEBaseName.split(',')
    # print('satSystems = %s' % satSystems)
    TLEFileNames = []
    for i, satSyst in enumerate(satSystems):
        url = 'http://www.celestrak.com/NORAD/elements/%s.txt' % satSyst
        # print('url = %s' % url)
        TLEFileNames.append(url.split('/')[-1])
        # print('TLEFileNames = %s' % TLEFileNames[-1])

        # sys.exit(0)
        # NOTE the stream=True parameter
        try:
            r = requests.get(url, stream=True)
            with open(TLEFileNames[-1], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                        # f.flush() commented by recommendation from J.F.Sebastian
                        os.fsync(f)
        except requests.exceptions.ConnectionError:
            sys.stderr.write('Connection to NORAD could not be established.\n')

            # check if we have alocal TLE file
            if os.path.isfile(TLEFileNames[-1]):
                print('Using local file %s' % TLEFileNames[-1])
                return TLEFileNames[-1]
            else:
                sys.stderr.write('Program exits.\n')
                sys.exit(E_REQUEST_ERROR)

    # catenate into a single TLE file that combines the different satellite systems
    if np.size(TLEFileNames) > 1:
        outFileName = satSystem.replace(',', '-') + '.txt'
        # print('outFileName = %s' % outFileName)
        fout = open(outFileName, 'w')

        fin = fileinput.input(files=TLEFileNames)
        for line in fin:
            fout.write(line)
        fin.close()
    else:
        outFileName = TLEFileNames[0]

    # print('outFileName = %s' % outFileName)
    if verbose:
        print('  TLEs saved in %s' % outFileName)

    return outFileName


def createDOPFile(observer, satSystem, listSat, predDates, xDOPs, cutoff, verbose=False):
    '''
    createDOPFile writes info to a file
    Parameters:
        observer: info about the observation station and date
        satSystem: used satellite system
        listSat: list of satellites
        predDates: contains the prediction dates
        xDOPs: the HDOP, VDOP and TDOP in that order
    '''
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d-DOP.txt' % (observer.getYMD())
    if verbose:
        print('  Creating DOP file: %s' % filename)
    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d' % observer.getYMD())
        fid.write('  cutoff: %2d\n\n' % cutoff)

        fid.write('      |#Vis|   HDOP   VDOP   PDOP   TDOP   GDOP\n\n')
        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # number of visible satellites
            fid.write(' | %2d |' % np.count_nonzero(~np.isnan(elev[i, :])))

            # write the DOP values in order
            if ~np.isnan(xDOPs[i, 0]):
                PDOP2 = xDOPs[i, 0] * xDOPs[i, 0] + xDOPs[i, 1] * xDOPs[i, 1]
                fid.write(' %6.1f %6.1f %6.1f %6.1f %6.1f' % (xDOPs[i, 0], xDOPs[i, 1], np.sqrt(PDOP2), xDOPs[i, 2], np.sqrt(PDOP2 + xDOPs[i, 2] * xDOPs[i, 2])))
            else:
                fid.write(' ------ ------ ------ ------ ------')
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def createGeodeticFile(observer, satSystem, listSats, predDates, lats, lons, verbose=False):
    '''
    createGeodeticFile creates a file containing lat/lon values for each satellite
    Parameters:
        observer: info about the observation station and date
        satSystem: used satellite system
        listSat: list of satellites
        predDates: contains the prediction dates
        lats/lons: latitude/longitude of all SVs
    '''
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d-GEOD.txt' % (observer.getYMD())
    if verbose:
        print('  Creating substellar file: %s' % filename)

    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d\n\n' % observer.getYMD())

        # write the sat IDs on first line
        satLine1 = ''
        satLine2 = ''
        for j, sat in enumerate(listSats):
            if len(sat.name) < 11:
                satLine1 += '  %10s' % sat.name
            else:
                satLine1 += '  %10s  ' % sat.name[:10]
                endChar = min(20, len(sat.name))
                satLine2 += '  %10s  ' % sat.name[10:endChar]
        fid.write('      %s' % satLine1)
        fid.write('\n')
        if len(satLine2) > 0:
            fid.write('      %s' % satLine2)
            fid.write('\n')
        fid.write('\n')

        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # write the lat/lon values
            for j, sat in enumerate(listSats):
                fid.write("  %5.1f %6.1f" % (lats[i, j], lons[i, j]))
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def createVisibleSatsFile(observer, satSystem, listSat, predDates, elevation, azimuth, cutoff, verbose=False):
    '''
    createVisibleSatsFile writes info to a file
    Parameters:
        observer: info about the observation station and date
        satSystem: used satellite system
        listSat: list of satellites
        predDates: contains the prediction dates
        elevation, azimuth contain the angles (elevation only if greater than cutoff angle)
    '''
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d.txt' % (observer.getYMD())
    if verbose:
        print('  Creating visibility file: %s' % filename)

    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d' % observer.getYMD())
        fid.write('  cutoff: %2d\n\n' % cutoff)

        # write the sat IDs on first line
        satLine1 = ''
        satLine2 = ''
        for j, sat in enumerate(listSat):
            if len(sat.name) < 11:
                satLine1 += '  %10s' % sat.name
            else:
                satLine1 += '  %10s' % sat.name[:10]
                endChar = min(20, len(sat.name))
                satLine2 += '  %10s' % sat.name[10:endChar]
        fid.write('      |#Vis|%s' % satLine1)
        fid.write('\n')
        if len(satLine2) > 0:
            fid.write('            %s' % satLine2)
            fid.write('\n')
        fid.write('\n')

        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # number of visible satellites
            fid.write(' | %2d |' % np.count_nonzero(~np.isnan(elevation[i, :])))

            for j, sat in enumerate(listSat):
                if math.isnan(elevation[i, j]):
                    fid.write('  ---- -----')
                else:
                    fid.write('  %4.1f %5.1f' % (elevation[i, j], azimuth[i, j]))
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def plotVisibleSats(systSat, observer, listSats, predDates, elev, cutoff, verbose=False):
    '''
    plotVisibleSats plots the timeline of visible satellites
    Parameters:
        systSat: satellite systems
        observer: info about the observer
        listSats: list of satellites
        predDates: prediction times
        elev: elevation angle (NaN if smaller than cutoff)
        cutoff: cutoff elevation (in degrees)
    '''
    plt.style.use('BEGPIOS')

    fig = plt.figure(figsize=(20.0, 16.0))
    # plt.subplots_adjust(top=0.65)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.97])
    # plt.tight_layout(fig, rect=[0, 0.03, 1, 0.97])
    ax1 = plt.gca()

    # ax2 = ax1.twinx()
    # set colormap
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))

    # plot the lines for visible satellites

    for i, sat in enumerate(listSats):
        elev2 = copy.deepcopy(elev)
        elev2[~np.isnan(elev2)] = i + 1
        plt.plot(predDates, elev2[:, i], linewidth=5, color=next(colors), label=sat.name)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=0, interval=1, tz=None))

    # create array for setting the satellite labels on y axis
    satNames = []
    for i in range(len(listSats)):
        # print('%s' % listSats[i].name)
        satNames.append(listSats[i].name)

    # set the tick marks
    plt.xticks(rotation=50, size='medium')
    plt.yticks(range(1, len(listSats) + 1), satNames, size='small')

    # color the sat labels ticks
    colors2 = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    for i, tl in enumerate(ax1.get_yticklabels()):
        tl.set_color(next(colors2))

    plt.grid(True)
    # set the limits for the y-axis
    plt.ylim(0, len(listSats) + 2)
    ax1.set_xlabel('Time of Day', fontsize='x-large')
    # plot title
    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large')
    yyyy, mm, dd = observer.getYMD()
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - Cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes, fontsize='medium')
    # plt.title('Station: %s @ %s, %s date %04d/%02d/%02d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd))

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-visibility.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def plotSkyView(systSat, observer, listSats, predDates, elevations, azimuths, curoff, verbose=False):
    '''
    plotSkyView plots the skyview for current location
    Parameters:
        systSat: satellite systems
        observer: info about the observer
        listSats: list of satellites
        predDates: prediction times
        elev, azim: elevation/azim angle (NaN if smaller than cutoff)
        cutoff: cutoff angle used
    '''
    plt.style.use('BEGPIOS')

    # rc('grid', color='#999999', linewidth=1, linestyle='-', alpha=[0].6)
    rc('xtick', labelsize='x-small')
    rc('ytick', labelsize='x-small')

    # force square figure and square axes looks better for polar, IMO
    width, height = rcParams['figure.figsize']
    size = min(width, height) * 2

    # make a square figure
    fig = figure(figsize=(size, size))

    # set the axis (0 azimuth is North direction, azimuth indirect angle)
    ax = fig.add_axes([0.10, 0.15, 0.8, 0.8], projection=u'polar')  # , axisbg='#CCCCCC', alpha=0.6)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Define the xticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, 13))
    xLabel = ['N', '30', '60', 'E', '120', '150', 'S', '210', '240', 'W', '300', '330']
    ax.set_xticklabels(xLabel)

    # Define the yticks
    ax.set_yticks(np.linspace(0, 90, 7))
    yLabel = ['', '75', '60', '45', '30', '15', '']
    ax.set_yticklabels(yLabel)

    # draw a grid
    grid(True)

    # plot the skytracks for each PRN
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    satLabel = []
    # print('elevations = %s' % elevations)
    # print('#listSats = %d' % np.size(listSats))

    # find full hours in date to set the elev/azimaccordingly
    # indexHour = np.where(np.fmod(prnTime, 3600.) == 0)
    # print('predDates = %s' % predDates[0].time())
    # print('predDates = %s  = %s' % (predDates[0].time(), hms_to_seconds(predDates[0].time())))
    predTimeSeconds = []
    hourTxt = []

    for t, predDate in enumerate(predDates):
        predTimeSeconds.append(hms_to_seconds(predDate.time()))
    predTimeSeconds = np.array(predTimeSeconds)
    # print('predTimeSeconds = %s' % predTimeSeconds)

    indexHour = np.where(np.fmod(predTimeSeconds, 3600.) == 0)
    # print('indexHour = %s' % indexHour)
    hourTxt.append(predTimeSeconds[indexHour])
    # print('hourTxt = %s' % hourTxt)

    for i, prn in enumerate(listSats):
        satLabel.append('%s' % prn.name)
        satColor = next(colors)
        azims = [np.radians(az) for az in azimuths[:, i]]
        elevs = [(90 - el) for el in elevations[:, i]]
        # print('PRN = %s' % prn.name)
        # print('elev = %s' % elevs)
        # print('azim = %s' % azims)
        # ax.plot(azims, elevs, color=next(colors), linewidth=0.35, alpha=0.85, label=satLabel[-1])
        ax.plot(azims, elevs, color=satColor, marker='.', markersize=3, linestyle='-', linewidth=1, label=satLabel[-1])

        # annotate with the hour labels
        prnHourAzim = azimuths[:, i][indexHour]
        # print('azimuth     = %s' % azimuths[:, i])
        # print('prnHourAzim = %s\n' % prnHourAzim)
        prnHourElev = elevations[:, i][indexHour]
        # print('Elevuth     = %s' % elevations[:, i])
        # print('prnHourElev = %s\n\n' % prnHourElev)

        hrAzims = [np.radians(az + 2) for az in prnHourAzim]
        hrElevs = [(90 - el) for el in prnHourElev]
        # print('hrAzims = %s' % hrAzims)
        # print('hrElevs = %s' % hrElevs)
        # print('-' * 20)
        # print('hourTxt = %s' % hourTxt)
        for j, hr in enumerate(hourTxt[0]):
            hrEl = hrElevs[j]
            if ~np.isnan(hrEl):
                hrAz = hrAzims[j]
                # print('hr = %s' % hr)
                # print('hrEl = %s' % hrEl)
                # print('hrAz = %d' % hrAz)
                hr = int(float(hr) / 3600.)
                # print('hr = %s' % hr)
                # print('hrEl = %d' % hrEl)
                plt.text(hrAz, hrEl, hr, fontsize='x-small', color=satColor)
        # print('-' * 30)

    # adjust the legend location
    mLeg = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=min(np.size(satLabel), 5), fontsize='small', markerscale=4)
    for legobj in mLeg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large', x=0.5, y=0.99, horizontalalignment='center')
    yyyy, mm, dd = observer.getYMD()
    # annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - Cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    # plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize='x-large')
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s)' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon)))
    plt.text(-0.075, 0.975, annotateTxt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize='medium')
    annotateTxt = (r'Date %04d/%02d/%02d - Cutoff %2d' % (yyyy, mm, dd, cutoff))
    plt.text(-0.075, 0.950, annotateTxt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize='medium')

    # needed for having radial axis span from 0 => 90 degrees and y-labels along north axis
    ax.set_rmax(90)
    ax.set_rmin(0)
    ax.set_rlabel_position(0)

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-skyview.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def hms_to_seconds(t):
    # print('t.hour %s' % t.hour)
    # h, m, s = [int(i) for i in t.split(':')]
    return 3600 * t.hour + 60 * t.minute + t.second


# def steppify(arr,isX=False,interval=0):
#     """
#     Converts an array to double-length for step plotting
#     """
#     if isX and interval==0:
#         interval = abs(arr[1]-arr[0]) / 2.0
#         newarr = array(zip(arr-interval,arr+interval)).ravel()
#         return newarr


def plotSatTracks(systSat, observer, listSats, predDates, satLats, satLons, verbose=False):
    '''
    plotSatTracks plots the ground tracks of the satellites on a map
    Parameters:
        systSat: satellite systems
        observer: info about the observer
        listSats: list of satellites
    '''
    plt.style.use('BEGPIOS')
    plt.figure(figsize=(16.0, 10.5))

    # miller projection
    map = Basemap(projection='mill', lon_0=0)
    # plot coastlines, draw label meridians and parallels.
    map.drawcoastlines()
    map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])
    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    map.drawmapboundary(fill_color='whitesmoke')
    map.fillcontinents(color='lightgray', lake_color='whitesmoke', alpha=0.9)

    # plot the baseStation on the map
    xObs, yObs = map(observer.lon / ephem.pi * 180., observer.lat / ephem.pi * 180.)
    # print('observer = %s %s' % (observer.lat, observer.lon))
    # print('observer = %s %s' % (ephem.degrees(observer.lat), ephem.degrees(observer.lon)))
    # print('observer = %f %f' % (observer.lat / ephem.pi * 180., observer.lon / ephem.pi * 180.))
    # print('xObs,yObs = %f  %f' % (xObs, yObs))
    map.plot(xObs, yObs, color='blue', marker='o', markersize=5)
    offSet = 0.5  # 1/10 of a degree
    xObs, yObs = map(observer.lon / ephem.pi * 180. + offSet, observer.lat / ephem.pi * 180. + offSet)
    plt.text(xObs, yObs, observer.name, fontsize='small', color='blue')

    # set colormap
    # colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    satLabel = []
    for i, SV in enumerate(listSats):
        satLabel.append('%s' % SV.name)
        # print('\n\nSat %s' % satLabel[-1])
        satColor = next(colors)

        # for j, dt in enumerate(predDates):
        # check whether we have a jump bigger than 180 degreein longitude
        lonDiffs = np.abs(np.diff(satLons[:, i]))
        # print('lons = %s' % satLons[:, i])
        # print('lonDiffs = %s' % lonDiffs)

        # lonDiffMax = np.max(lonDiffs)
        # print('lonDiffMax = %s' % lonDiffMax)

        lonIndices = np.where(lonDiffs > 300)
        # print('lonIndices = %s' % lonIndices)

        # split up the arrays satLons and satLats based on the lonIndices found
        if np.size(lonIndices) > 0:
            for k, lonIndex in enumerate(lonIndices[0]):
                # print('lonIndex = %s' % lonIndex)
                # print('lonIndex[%d] = %d  satLons[%d] = %f' % (k, lonIndex, lonIndex, satLons[lonIndex, i]))

                # determine indices between which we have a track without 360 degree jump
                if k == 0:
                    startIndex = 0
                else:
                    startIndex = lonIndices[0][k - 1] + 1
                endIndex = lonIndex + 1

                xSat = np.zeros(np.size(predDates))
                ySat = np.zeros(np.size(predDates))
                xSat.fill(np.nan)
                ySat.fill(np.nan)

                # print('startIndex = %d  endIndex = %d' % (startIndex, endIndex))

                for l in range(startIndex, endIndex):
                    xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                    # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

                # print('intermed x = %s' % xSat)
                map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle='-', marker='.', markersize=6)

            xSat = np.zeros(np.size(predDates))
            ySat = np.zeros(np.size(predDates))
            xSat.fill(np.nan)
            ySat.fill(np.nan)
            for l in range(lonIndex + 1, np.size(predDates)):
                xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

            # print('last part x = %s' % xSat)
            map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle='-', marker='.', markersize=6, label=satLabel[-1])
        else:
            xSat = np.zeros(np.size(predDates))
            ySat = np.zeros(np.size(predDates))
            xSat.fill(np.nan)
            ySat.fill(np.nan)

            for l in range(np.size(predDates)):
                xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

            # print('full part x = %s' % xSat)
            map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle='-', marker='.', markersize=6, label=satLabel[-1])

    # adjust the legend location
    mLeg = plt.legend(bbox_to_anchor=(0.5, 0.05), loc='lower center', ncol=min(np.size(satLabel), 5), fontsize='small', markerscale=2)
    for legobj in mLeg.legendHandles:
        legobj.set_linewidth(5.0)

    # plot title
    yyyy, mm, dd = observer.getYMD()
    plt.title(('%s Satellite Groundtracks - Date %04d/%02d/%02d' % (systSat.replace(',', ' & ').upper(), yyyy, mm, dd)), fontsize='x-large')

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-groundtrack.png' % (observer.getYMD())
    plt.savefig(filename)

    if verbose:
        plt.show()


def plotDOPVisSats(systSat, observer, listSats, predDates, elev, xDOPs, cutoff, verbose=False):
    '''
    plotDOPVisSats plots the xDOP values and the total number of satellites visible
    Parameters:
        systSat: satellite systems
        observer: info about the observer
        listSats: list of satellites
        predDates: prediction times
        elev: elevation angle (NaN if smaller than cutoff), used for determining the number of visible satellites
        xDOPs: the xDOP values HDOP, VDOP and TDOP
    '''
    plt.style.use('BEGPIOS')

    fig = plt.figure(figsize=(20.0, 16.0))
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # second y-axis needed, so make the x-axis twins
    # set colormap

    # plot the number of visible satellites
    nrVisSats = []
    for i, el in enumerate(elev):
        nrVisSats.append(np.count_nonzero(~np.isnan(el)))
        ax2.set_ylim(0, max(nrVisSats) + 1)
    # print('nrVisSats = %s' % nrVisSats)
    ax2.plot(predDates, nrVisSats, linewidth=3, color='black', drawstyle='steps-post', label='#Visible')
    # ax2.fill_between(steppify(predDates,isX=True), steppify(nrVisSats)*0, steppify(nrVisSats), facecolor='b',alpha=0.2)
    # ax2.fill_between(predDates, 0, nrVisSats, color='lightgray', alpha=0.5, drawstyle='steps')
    # ax2.fill_between(lines[0].get_xdata(orig=False), 0, lines[0].get_ydata(orig=False))

    # plot the xDOPS on first axis
    ax1.set_ylim(0, maxDOP)
    # print('len(xDOPs) = %d' % len(xDOPs[0, :]))
    colors = iter(cm.jet(np.linspace(0, 1, len(xDOPs[0, :]) + 2)))
    # print('len(xDOPs[0, :]+2 = %s' % (len(xDOPs[0, :]) + 2))
    # print('colors.size = %s' % np.linspace(0, 1, len(xDOPs[0, :]) + 2))
    labels = ['HDOP', 'VDOP', 'TDOP']
    # print('labels = %s' % labels)
    for i in range(0, 3):
        xDOP = xDOPs[:, i]
        dopColor = colors.next()
        transparency = .5 - i * 0.1
        ax1.fill_between(predDates, 0, xDOP, color=dopColor, alpha=transparency)
        ax1.plot(predDates, xDOP, linewidth=2, color=dopColor, label=labels[i])

        # add PDOP
        if i is 1:
            PDOP2 = xDOPs[:, 0] * xDOPs[:, 0] + xDOPs[:, 1] * xDOPs[:, 1]
            # print('PDOP = %s' % np.sqrt(PDOP2))
            dopColor = colors.next()
            transparency = .2
            ax1.fill_between(predDates, 0, np.sqrt(PDOP2), color=dopColor, alpha=transparency)
            ax1.plot(predDates, np.sqrt(PDOP2), linewidth=2, color=dopColor, label='PDOP')

        # add GDOP
        if i is 2:
            GDOP = np.sqrt(PDOP2 + xDOPs[:, 2] * xDOPs[:, 2])
            # print('GDOP = %s' % GDOP)
            dopColor = colors.next()
            transparency = .1
            ax1.fill_between(predDates, 0, GDOP, color=dopColor, alpha=transparency)
            ax1.plot(predDates, GDOP, linewidth=2, color=dopColor, label='GDOP')

    ax1.legend(loc='upper left', frameon=True)

    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large')
    yyyy, mm, dd = observer.getYMD()
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes, fontsize='medium')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=0, interval=1, tz=None))

    plt.xticks(rotation=50, size='medium')
    ax1.set_ylabel('DOP [-]', fontsize='x-large')
    ax2.set_ylabel('# Visible satellites [-]', fontsize='x-large')
    ax1.set_xlabel('Time of Day', fontsize='x-large')

    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-DOP.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def treatCmdOpts(argv):
    '''
    treatCmdOpts treats the command line arguments using
    '''
    helpTxt = os.path.basename(__file__) + ' predicts the GNSS orbits based on TLEs'

    # create the parser for command line arguments
    parser = argparse.ArgumentParser(description=helpTxt)
    parser.add_argument('-s', '--satSystem', help='Name of satellite system in comma separated list (cfr NORAD naming)', required=True)
    parser.add_argument('-i', '--interval', help='interval in minutes (defaults to 20)', type=int, required=False, default=20)
    parser.add_argument('-c', '--cutoff', help='cutoff angle in degrees (defaults to 10)', type=int, required=False, default=10)
    parser.add_argument('-o', '--observer', help='Station info "name,latitude,longitude" (units = degrees, defaults to RMA)', required=False, default=None)
    parser.add_argument('-d', '--date', help='Enter prediction date (YYYY/MM/DD), defaults to today', required=False)
    parser.add_argument('-b', '--startTime', help='Enter start time (hh:mm), defaults to 00:00', required=False, default='00:00')
    parser.add_argument('-e', '--endTime', help='Enter end time (hh:mm), defaults to 23:59', required=False, default='23:59')
    parser.add_argument('-m', '--maxDOP', help='Maximum xDOP value to display, defaults to 15', required=False, default=15, type=int)
    parser.add_argument('-v', '--verbose', help='displays interactive graphs and increase output verbosity (default False)', action='store_true', required=False)

    args = parser.parse_args()

    return args.satSystem, args.interval, args.cutoff, args.observer, args.date, args.startTime, args.endTime, args.maxDOP, args.verbose


# main starts here
if __name__ == "__main__":
    # init defaullt station
    RMA = Station()
    RMA.init('RMA', '50:50:38.4551', '4:23:34.5421', ephem.date(ephem.now()))
    BERTRIX = Station()
    BERTRIX.init('BERTRIX', '49.894275', '5.241417', ephem.date(ephem.now()))
    # print('RMA = %s | %.9f  %.9f | %s  %s' % (RMA.name, RMA.lat, RMA.lon, ephem.degrees(RMA.lat), ephem.degrees(RMA.lon)))

    # treat the command line options
    satSystem, interval, cutoff, observer, predDate, startTime, endTime, maxDOP, verbose = treatCmdOpts(sys.argv)
    # print('satSystem = %s' % satSystem)
    # print('predDate = %s' % predDate)
    # print('startTime = %s' % startTime)
    # print('endTime = %s' % endTime)
    # print('interval = %s' % interval)
    # print('observer = %s' % observer)
    # print('cutoff = %d' % cutoff)
    # print('verbose = %s\n' % verbose)

    # import tle data from NORAD if internet_on(), save as sat=ephem.readtle(...)-----
    # TLEfile = getTLEfromNORAD(satSystem)
    TLEfile = getTLEfromNORAD(satSystem)
    # TLEfile = 'galileo.txt'
    # existTLEFile(folder, TLEfile, verbose)

    # read in the observer info (name, latitude, longitude, date
    obsStation = setObserverData(observer, predDate, verbose)
    if verbose:
        obsStation.statPrint()

    # read in the list of satellites from the TLE
    satList = loadTLE(TLEfile, verbose)

    # calculate the interval settings for a full day prediction starting at 00:00:00 hr of predDate
    predDateTimes, nrPredictions = setObservationTimes(obsStation, startTime, endTime, interval, verbose)

    # calculate the informations for each SVs in satList
    subLat = np.empty([nrPredictions, np.size(satList)])
    subLon = np.empty([nrPredictions, np.size(satList)])
    azim = np.empty([nrPredictions, np.size(satList)])
    elev = np.empty([nrPredictions, np.size(satList)])
    dist = np.empty([nrPredictions, np.size(satList)])
    dist_velocity = np.empty([nrPredictions, np.size(satList)])
    eclipsed = np.empty([nrPredictions, np.size(satList)])
    xDOP = np.empty([nrPredictions, 3])  # order is HDOP, VDOP, TDOP

    for i, dt in enumerate(predDateTimes):
        obsStation.date = dt
        elevTxt = ''
        for j, sat in enumerate(satList):
            sat.compute(obsStation)
            # print('sat[%d] = %26s   %5.1f   %4.1f' % (j, sat.name, np.rad2deg(sat.az), np.rad2deg(sat.alt)))
            subLat[i, j] = np.rad2deg(sat.sublat)
            subLon[i, j] = np.rad2deg(sat.sublong)
            azim[i, j] = np.rad2deg(sat.az)
            elev[i, j] = np.rad2deg(sat.alt)
            dist[i, j] = sat.range
            dist_velocity[i, j] = sat.range_velocity
            eclipsed[i, j] = sat.eclipsed

            # elevTxt += "%6.1f  " % elev[i][j]

        # determine the visible satellites at this instance
        indexVisSats = np.where(elev[i, :] >= cutoff)
        # print('indexVisSats = %s (%d)' % (indexVisSats, np.size(indexVisSats)))
        # print('elev = %s' % elev[i, :])
        # print('azim = %s' % azim[i,:])
        # print('elevRad = %s' % np.radians(elev[i,:]))

        # calculate xDOP values when at least 4 sats are visible above cutoff angle
        if np.size(indexVisSats) >= 4:
            A = np.matrix(np.empty([np.size(indexVisSats), 4], dtype=float))
            # print('A  = %s' % A)
            # print('type A = %s  ' % type(A))
            elevVisSatsRad = np.radians(elev[i, indexVisSats])
            azimVisSatsRad = np.radians(azim[i, indexVisSats])
            # print('elevVisSatsRad = %s' % elevVisSatsRad)
            # print('azimVisSatsRad = %s' % azimVisSatsRad)

            for j in range(np.size(indexVisSats)):
                A[j, 0] = np.cos(azimVisSatsRad[0, j]) * np.cos(elevVisSatsRad[0, j])
                A[j, 1] = np.sin(azimVisSatsRad[0, j]) * np.cos(elevVisSatsRad[0, j])
                A[j, 2] = np.sin(elevVisSatsRad[0, j])
                A[j, 3] = 1.
                # print('A[%d] = %s' % (j, A[j]))

            # calculate ATAInv en get the respective xDOP parameters (HDOP, VDOP and TDOP)
            AT = A.getT()
            ATA = AT * A
            ATAInv = ATA.getI()
            # print('AT = %s' % AT)
            # print('ATA = %s' % ATA)
            # print('ATAInv = %s' % ATAInv)

            xDOP[i, 0] = np.sqrt(ATAInv[0, 0] + ATAInv[1, 1])  # HDOP
            xDOP[i, 1] = np.sqrt(ATAInv[2, 2])  # VDOP
            xDOP[i, 2] = np.sqrt(ATAInv[3, 3])  # TDOP
        else:  # not enough visible satellites
            xDOP[i] = [np.nan, np.nan, np.nan]

        # print('xDOP[%d] = %s' % (i, xDOP[i]))

        # print('dt = %s' % dt)
        # print('lat= %s' % subLat[i][0])
        # print('lon= %s' % subLon[i][0])
        # print('az = %s' % azim[i][0])
        # print('el = %s' % elev[i][0])
        # print('r  = %s' % dist[i][0])
        # print('rV = %s' % dist_velocity[i][0])
        # print('ec = %s' % eclipsed[i][0])

        # print('%02d:%02d    %s' % (dt.hour, dt.minute, elevTxt))

    # # create index for satellites above cutoff angle
    # for j, sat in enumerate(satList):
    #     # print('elev[:][%d] = %s' % (j, elev[:, j]))
    #     indexVisibleSats = np.where(elev[:,j] >= cutoff)
    #     print('indexVisibleSats = %s (%d)' % (indexVisibleSats, np.size(indexVisibleSats)))
    # print('elev = %s' % elev[:,j])

    # set all elev < cutoff to NAN
    elev[elev < cutoff] = np.nan
    # print('elev = %s' % elev)

    # write to results file
    createVisibleSatsFile(obsStation, satSystem, satList, predDateTimes, elev, azim, cutoff, verbose)
    createDOPFile(obsStation, satSystem, satList, predDateTimes, xDOP, cutoff, verbose)
    createGeodeticFile(obsStation, satSystem, satList, predDateTimes, subLat, subLon, verbose)

    # create plots
    plotVisibleSats(satSystem, obsStation, satList, predDateTimes, elev, cutoff, verbose)
    plotDOPVisSats(satSystem, obsStation, satList, predDateTimes, elev, xDOP, cutoff, verbose)
    plotSkyView(satSystem, obsStation, satList, predDateTimes, elev, azim, cutoff, verbose)
    plotSatTracks(satSystem, obsStation, satList, predDateTimes, subLat, subLon, verbose)

    # show all plots
    if verbose:
        plt.show()

    # end program
    sys.exit(E_SUCCESS)
