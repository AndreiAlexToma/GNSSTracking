def download_tle():
    """ downloads the "ALL_TLE" file from the tle info server """

    if not os.path.isfile("ALL_TLE.TXT"):
        myfile = urllib.URLopener()
        myfile.retrieve('http://www.tle.info/data/ALL_TLE.ZIP', 'ALL_TLE.ZIP')
        azip = zipfile.ZipFile('ALL_TLE.ZIP')
        azip.extractall('.')
        print("TLE data obtained!")

    # load TLE data into python array
    with open('ALL_TLE.TXT') as f:
        tle_content = f.readlines()
        tle_content = [line.replace('\n', '') for line in tle_content]  # remove end lines
        print("loaded {0} TLEs".format(int(len(tle_content) /  3)))

    return tle_content