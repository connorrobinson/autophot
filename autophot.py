import numpy as np
import os
from astropy.io import fits
from astropy import wcs
import pdb
import matplotlib.pyplot as plt
import scipy.optimize as opt



def phot(file, path, gain, readnoise, ra = 0, dec = 0,range = 50, annulus = [150,3], forceAperture  = 0, boxsize = 300, plot = 0, fitgauss = True):
    '''
    autophot.phot
    
    !!!!!IMPORTANT!!!!!
    IF YOU ARE USING THE AUTOMATIC VERSION (ENTERING RA AND DEC), THIS MUST BE RUN IN PYTHON 2.X DUE TO USE OF ASTRONOMETRY.NET
    ASTROMETRY.NET MUST BE INSTALLED CORRECTLY FOR THIS TO WORK (i.e. callable from the command line)
    !!!!!IMPORTANT!!!!!
    
    PURPOSE: Performs aperture photometry with support from astrometry.net on a point source given RA and Dec
    
    INPUT:
        file: [string] Contains the file name
        path: [string] path to said file
    
    OPTIONAL INPUTS:
        ra: [float] right ascension of target, given in fractional hours
        dec: [float] declination of target, given in fractional degrees
        range: [integer] largest possible aperture radius in pixels, default is 20 pixels
        forceAperture: [integer] Force the code to use a specific aperture (in pixels)
        boxsize: [integer] length of one of the sides of the box that will be trimmed from the whole image for photometry
                 Should be larger than range
        manual: [boolean] Turn this on 
    
    OUTPUT:
        Array containing the flux in e- and the optimum aperture radius if the forceAperture keyword is not used
    
    
    NOTES:
        The manual clicks generate some text. Does not seem cause any issues.
    
    THINGS TO ADD IN THE FUTURE:
    
    1) Add a way to check that you're only getting counts from a single star
         (Kind of already done? If you're worried about this, turn plot on)
    
    '''

    
    #If both ra and dec are given, try to solve WCS solution with astrometry.net
    if ra != 0 and dec != 0:
        
        #Get the astrometry data for the image
        astrometry(file,path)
        
        #Open the new image with the WCS solution
        try:
            image = fits.open(path+str.split(file, '.fits')[0]+'.new')
            
            #Get the location of the target from the WCS solution
            w = wcs.WCS(image[0].header)
            pix = np.floor(np.array(w.wcs_world2pix(ra,dec, 1)))
            
        #If the file is missing, then try with manual photometry
        except IOError:
            print('WARNING: Missing .new file from astrometry')
            print('Continue with manual extraction? (y/n)')
            choice = raw_input()
            if choice == 'y' or choice == 'Y':
                image = fits.open(path+file)
                pix = select(image)
            else:
                return
    
    #If no ra and dec are given, proceed with manual extraction
    else:
        image = fits.open(path+file)
        pix = select(image)
    
    
    #Check to see if star is near the edge of the frame, reduce the maximum range if it is.
    if pix[0] - boxsize < 0 or pix[1]-boxsize < 0:
        print('Star too close/off the edge of frame! Extraction failed, check coordinates or try a smaller boxsize')
        return
    
    #Trim image into a box with lengths of boxsize
    box = image[0].data[pix[1]-boxsize/2:pix[1]+boxsize/2,pix[0]-boxsize/2:pix[0]+boxsize/2]
    
    #If the fitgauss flag is on, fit a gaussian to the star to get better centered coordinates
    if fitgauss == 1:
        fit = gaussfit(box)
        pix = np.floor([fit[1],fit[2]])
    
    #Build a euclidean distance array centered at the target
    xvect = np.arange(0,boxsize)-pix[0]
    xgrid = np.tile(xvect, (boxsize,1))
    yvect = np.arange(0,boxsize)-pix[1]
    ygrid = np.transpose(np.tile(yvect, (boxsize,1)))
    euc = np.sqrt(xgrid**2 + ygrid**2)
    
    #Extract the median sky counts from an annulus
    skyannulus = box[(euc >= annulus[0]) &  (euc <= annulus[0]+annulus[1])]
    sky = np.median(skyannulus)
    
    #Begin extracting counts
    signal = np.array([])
    SNR = np.array([])
    radarr = np.arange(range)+1
    apersum = np.array([])
    
    #If forcing the aperture to be fixed, calculate flux using that aperture
    if forceAperture != 0:
        aper = box[(euc <= forceAperture)]
        apersum = np.append(apersum,np.sum(aper))
        aperpix = len(aper)
        signal = np.append(signal,apersum - sky*aperpix)
        
    else:
        for i, radius in enumerate(radarr):
            #Define region smaller than r
            aper = box[(euc <= radius)]
            apersum = np.append(apersum,np.sum(aper))
            aperpix = len(aper)
            
            #Subtract sky counts from signal, weighted by aperture size
            signal = np.append(signal,apersum[i] - sky*aperpix)
            
            #Calculate sky counts in e- in the aperture
            Nsky = gain*aperpix*sky
            
            #Calculate SNR
            SNR = np.append(SNR,gain*signal[i]/np.sqrt(gain*signal[i] + Nsky + aperpix*readnoise**2))
            
    #If the plot keyword is turned on, show a plot of SNR vs. aperture radius
    if plot == 1:
        
        #Plot the image with the selected star circled
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)
        
        ax.scatter(pix[0],pix[1],s = 200, alpha = .5, c = 'b')
        ax.imshow(np.log10(box), cmap = 'cubehelix', origin = 'lower')
        plt.show()
        
        #Make a plot of SNR
        if forceAperture == 0:
            plt.plot(SNR)
            plt.xlabel('Aperture Radius in pixels')
            plt.ylabel('Signal to noise')
            plt.show()
        
    #Grab the radius where SNR is maximized
    if forceAperture == 0:
        maxind = np.argmax(SNR)
        maxrad = radarr[maxind]
        flux = apersum[maxind]*gain
        return flux, maxrad
        
    else:
        flux = apersum[0] * gain
        return flux

def astrometry(file, path):
    '''
    autophot.astrometry
    
    PURPOSE: Calls astrometry.net to solve for a WCS solution
    
    INPUTS:
        file: [string] Contains file name, should be a fits image
        path: [string] Path to the location of file
    
    OUTPUTS:
        Does not return anything
        Does produce files from doing th astrometry
    
    '''
    
    os.system('solve-field --downsample 4 --overwrite --no-plots --guess-scale --no-fits2fits '+path+file)
    

def gauss2d((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    autophot.gauss2d
    
    PURPOSE: Creates a 2d gaussian for fitting stellar PSFs
    
    INPUT:
        A: [float] Amplitude
        x: [float] Array of x values
        y: [float] Array of y values
        x0: [float] Offset for X
        y0: [float] Offset for Y
        xsig: [float] Width of x component
        ysig: [float] Width of y component
        off: [float] Offset for entire array above zero
    
    OUTPUT:
        Array containing the gaussian calculated at each pair of x and y values
    '''
    
    #Define parameters and calculate gaussian
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    
    return g.ravel()
    

def gaussfit(image):
    '''
    autophot.gaussfit
    
    PURPOSE: Fits the star with a gaussian to get a better guess of the xy coordinates of the star
    
    INPUTS:
        image: [2d float array] image containing a single star
    
    NOTE:
        This will only be useful for single stars
    '''
    
    lenx = len(image[0,:])
    leny = len(image[:,0])
    
    #Set up x and y arrays
    xvec = np.linspace(0, lenx-1, lenx)
    yvec = np.linspace(0, leny-1, leny)
    
    x,y = np.meshgrid(xvec, yvec)
    
    #Set up initial guesses
    amplitude = 1000
    x0 = np.floor(lenx/2)
    y0 = np.floor(leny/2)
    xsig = 20
    ysig = 20
    theta = 0
    off  = 0 
    
    initial = (amplitude, x0,y0, xsig, ysig, theta, off)
    
    #Fit the gaussian
    popt, pcov = opt.curve_fit(gauss2d, (x,y), image.ravel(), p0 = initial)
    
    return popt

def select(image):
    """
    
    autophot.select
    
    PURPOSE: plots an image, click on star to get xy coords
    
    INPUTS:
        image: [HDU object] Image produced using fits.open 
    
    OUTPUTS: 
        XY coordinates of the selected star
    
    """
    
    #Set up global coordinates for x and y
    coords = [] 
    global coords
    
    #Create the figure
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.log10(image[0].data), cmap  = 'cubehelix', origin='lower')
    
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    plt.show()
    
    fig.canvas.mpl_disconnect(cid)
    return np.round(coords)

def onclick(event):
    '''
    autophot.onclick
    
    Used alongside select to click on a star to get xy coords
    '''
    print(np.round(event.xdata))
    print(np.round(event.ydata))
    
    x = event.xdata
    y = event.ydata

    
    global coords
    coords = [x, y]
    
    plt.close()
    return

