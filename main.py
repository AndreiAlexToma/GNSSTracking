# First, the required modules are imported. The array-manipulation module
# **numpy**, the `matplotlib` submodule **pyplot** and the map-plotting toolkit
# **basemap**. It is common practice to use the aliases `np` and `plt` for the
# corresponding modules.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Now the latitude and longitude coordinates are read from a
# csv file. It contains information about airports all around the world. Only
# the columns 6 and 7 are read, since they contain the relevant information
# for the plot.

airports = np.genfromtxt("airports.dat",
                         delimiter=',', 
                         dtype=[('lat', np.float32), ('lon', np.float32)], 
                         usecols=(6, 7))

fig = plt.figure()

# Now the map is created. It uses the *Gall Stereographic Projection* but there
# are [many others](http://matplotlib.org/basemap/users/mapsetup.html)
# available. The `resolution` is set to `l`ow, it is also possible to draw
# `i`ntermediate, `c`rude, `h`igh and `f`ull resolution maps. The parameter 
# `area_thresh` is the minimal area, in square kilometres, for a coastline or
# lake to be included in the map.
themap = Basemap(projection='gall',
              llcrnrlon = -15,              # lower-left corner longitude
              llcrnrlat = 28,               # lower-left corner latitude
              urcrnrlon = 45,               # upper-right corner longitude
              urcrnrlat = 73,               # upper-right corner latitude
              resolution = 'l',
              area_thresh = 100000.0,
              )
              
# The next lines will:
# * add the coastlines
# * add the country borders
# * fill the land in the `gainsboro` colour. Html hexadecimal notation and html
# colour names can be used here
# * set the oceans colour to `steelblue`
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')

# Now we compute the coordinates of the airports in the corresponding
# projection and add the points to the map. The last command will finally
# display the plot.
x, y = themap(airports['lon'], airports['lat'])
themap.plot(x, y, 
            'o',                    # marker shape
            color='Indigo',         # marker colour
            markersize=4            # marker size
            )

plt.show()
