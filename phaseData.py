import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
import sys

#*************************************************#
#-------------------------------------------------#
#    This is a short script that aims to          #
#    automate the free energy calculations,       #
#    given the data. See accompanying Jupyter     #
#    notebook to see how these defined methods    #
#    and objects are used.                        #
#                                                 #
#    Written by Irem Altan (2018)                 #
#    (contact irem.altan@gmail.com for questions) #
#-------------------------------------------------#
#*************************************************#

# find intersections between two curves---------------------------------
def findIntersections(iso1,iso2,tol=0.01):
    x1 = iso1.mu.x
    x2 = iso2.mu.x
    y1 = iso1.mu.y
    y2 = iso2.mu.y
    # first, modify functions such that x1 and x2 are identical
    startInd = max(min(x1),min(x2))
    endInd = min(max(x1),max(x2))
    xint = np.arange(startInd,endInd+tol,tol)
    cubicSpline1 = interpolate.splrep(x1,y1)
    cubicSpline2 = interpolate.splrep(x2,y2)
    y1new = interpolate.splev(xint, cubicSpline1, der=0)
    y2new = interpolate.splev(xint, cubicSpline2, der=0)
    
    # now get intersection (stackoverflow Q 28766692)
    idx = np.argwhere(np.diff(np.sign(y1new-y2new)) != 0).reshape(-1) + 0

    # get corresponding densities
    if iso1.eos.interpolant == None:
        iso1.eos.genInterpolant('linear')
    if iso2.eos.interpolant==None:
        iso2.eos.genInterpolant('linear')
    rho1 = iso1.eos.interpolant.getSinglePoint(xint[idx])
    rho2 = iso2.eos.interpolant.getSinglePoint(xint[idx])
    return xint[idx],y1new[idx],rho1,rho2
#-----------------------------------------------------------------------

# function that plots multiple chemical potentials----------------------
def plotMultipleMu(listOfIso, listOfLegendStrs = None, xaxis="Beta", yaxis="Beta*Mu"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for iso in listOfIso:
        ax.plot(iso.mu.x,iso.mu.y)

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
#     The data class defines data objects
#     that hold information about a phase.
#-----------------------------------------------------------------------         
class data():
    def __init__(self,x,y):
        # x & y are np.arrays that hold the data
        self.x = x
        self.y = y
        # define the range of data. Min to max by default
        self.startData = np.min(self.x)
        self.endData = np.max(self.x)

        # find indices of the values closest to the start and end values
        self.findNearest()

        # no interpolant object by default
        self.interpolant = None

    # method to change the data range (min to max by def)
    def setRange(self,a,b):
        self.startData = a
        self.endData = b

        # make sure interval is contained within the dataset
        if self.startData < self.x[0]:
            self.startData = self.x[0]
        if self.endData > self.x[-1]:
            self.endData = self.x[-1]
        # get indices of the start and end values
        self.findNearest()

    # method to plot the data using matplotlib
    def plot(self, displayFit=1, labelx='', labely='', noShow=0):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # scatter always plots all data, regardless of interval
        # this helps validate the interval chosen visually
        ax.scatter(self.x, self.y) 

        # label axes
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)

        # plot the interpolant if it exists
        if self.interpolant != None:
            ax.plot(self.interpolant.xint, self.interpolant.yint, 'r-')
        else:
            if displayFit==1:
                print("WARNING: interpolant does not exist")
        # show the plot
        plt.show()

    # method to interpolate data
    # 3rd degree polynomial by default
    def genInterpolant(self, method='poly', degree=3):
        self.interpolant = interpolant(self, method, degree)
    
    # find indices for the values closest to start and end values
    def findNearest(self):
        self.startIdx = (np.abs(self.x - self.startData)).argmin()
        self.endIdx = (np.abs(self.x - self.endData)).argmin()
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The data class defines interpolant objects
#     that hold information about a phase.
#-----------------------------------------------------------------------
class interpolant():
    def __init__(self, data, method, degree, tol=0.001):
        # method = poly and method = linear are acceptable
        # if method = linear, degree is ignored
        self.method = method
        self.popt = None

        # define x interval for interpolation
        self.xint = np.arange(data.startData, data.endData + tol, tol)

        # make sure xint is within the data range
        if self.xint[-1] > data.x[-1]:
            self.xint[-1] = data.x[-1]
            #self.xint = np.delete(self.xint,-1)
        
        # set degree of polynomial fit
        self.degree = int(degree)
        # polynomial fit object
        self.p = None
        # linear interpolation object
        self.linearInterpolation = None
        # starting and ending indices for fit
        s = data.startIdx
        e = data.endIdx+1

        # if poly fit is required
        if method == 'poly':
            self.p = np.polyfit(data.x[s:e], data.y[s:e], self.degree)
            self.yint = np.zeros(len(self.xint))

            for i in range(self.degree+1):
                self.yint += self.p[self.degree-i]*self.xint**i
        # or if linear interpolation is required
        elif method == 'linear':
            self.linearInterpolation = interpolate.interp1d(data.x, data.y, 'linear')
            self.yint = self.linearInterpolation(self.xint)
        elif method == 'inverse':
            def inv(x,a,b):
                return a/x+b
            self.popt, pcov = curve_fit(inv, data.x[s:e], data.y[s:e])
            self.yint = inv(self.xint, *self.popt)

    # method that interpolates a single point in the dataset
    def getSinglePoint(self, xQuery):
        # check is the fit is polynomial
        isPoly = 1
        
        # determine if a polynomial fit is expected
        try:
            isPoly = 0 if self.p == None else 1
        except:
            isPoly = 0 if None in self.p else 1
        
        # extract value
        if isPoly == 1:
            yQuery = 0.
            for i in range(self.degree+1):
                yQuery += self.p[self.degree-i]*xQuery**i
            if np.size(xQuery)>1:
                return xQuery, yQuery
            else:
                return yQuery
        elif self.linearInterpolation != None:
            if np.size(xQuery) > 1:
                largerThanOne = 1
            else:
                largerThanOne = 0
            if largerThanOne == 1:
                while xQuery[-1] > self.xint[-1]:
                    xQuery = np.delete(xQuery,-1)
                while xQuery[0] < self.xint[0]:
                    xQuery = np.delete(xQuery,0)
            else:
                if xQuery > self.xint[-1] or xQuery < self.xint[0]:
                    sys.exit("query point is outside of the region linearly interpolated")
            if largerThanOne == 1:
                return xQuery, self.linearInterpolation(xQuery)
            else:
                return self.linearInterpolation(xQuery)
        elif self.method == "inverse":
            def inv(x,a,b):
                return a/x+b
            yQuery = inv(xQuery, *self.popt)
            return xQuery, yQuery
        else:
            print("WARNING: no interpolant exists")
            return None

    # calculate a single integral
    def integrateSingle(self, a, b, tol=0.001):
        if a > b:
            a, b = b, a
        xint = np.arange(a,b+tol,tol)
        if np.size(xint) > 1:
            xintNew, yint = self.getSinglePoint(xint)
        else:
            yint = self.getSinglePoint(xint)
            xintNew = xint
        return np.trapz(yint,xintNew)
    
    # calculate the integral as a function
    def integralFunction(self, start, end, ref, tol=0.001):
        s = min(start, end)
        e = max(start, end)
        xint = np.arange(s,e+tol,tol)
        yint = np.zeros(xint.size)
        for i, val in enumerate(xint):
            sign = -1 if ref > val else 1
            yint[i] = sign * self.integrateSingle(ref, val)
        
        return xint, yint
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The phase class defines objects that
#     hold phase information.
#-----------------------------------------------------------------------
class phase():
    def __init__(self, fname, N, skipHeader=1):
        # read data from file
        self.rawData = np.genfromtxt(fname,skip_header=skipHeader)
        # set number of particles
        self.N = N
        # initialize the chemical potential object
        self.mu = None
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The ideal class defines the phase information for
#     integration from the ideal gas. (See eq. 10 in the chapter)
#-----------------------------------------------------------------------
class ideal(phase):
    def __init__(self, fname, N, beta, targetP, interp='poly',skipHeader=1):
        # read from file
        phase.__init__(self,fname,N,skipHeader)
        # set parameters
        self.beta = beta
        self.targetP = targetP
        # create data object for the equation of state (rho(P))
        self.eos = data(self.rawData[:,0], self.rawData[:,1])
        # generate interpolant (fits a 3rd order polynomial)
        self.eos.genInterpolant(interp,3)
        # set target density to be integrated to
        self.targetRho = self.eos.interpolant.getSinglePoint(self.targetP)
        # initialize the integrand object
        self.integrand = None
        self.B2 = None
        # organize the data
        self.rho = self.eos.y
        self.P = self.eos.x
        # get eos as pressure as a function of rho
        self.eosInv = data(self.rawData[:,1], self.rawData[:,0])
        self.eosInv.genInterpolant(interp,3)

    # plot equation of state
    def plotEos(self):
        self.eos.plot()
    
    # plot the integrand
    def plotIntegrand(self):
        self.integrand.plot()

    # calculate the integrand
    def calcIntegrand(self):
        integrand = self.beta*self.P/(self.rho**2) - 1./self.rho
        self.integrand = data(np.append(0, self.eos.y), np.append(self.B2, integrand))

    # calculate the chemical potential at a specific point (eq. 10)
    def calcMu(self):
        self.intVal = self.integrand.interpolant.integrateSingle(0.,self.targetRho)
        self.A = self.intVal + np.log(self.targetRho) - 1. + 1./self.N*np.log(2*np.pi*self.N)
        # beta*mu = beta*A/N + beta*P/rho
        self.mu = self.A + self.beta * self.targetP/self.targetRho 
    
    # calculate the chemical potential as a function of P
    def calcMuFunction(self):
        rho, integral = self.integrand.interpolant.integralFunction(0.05,self.integrand.endData,0.)
        rhonew, P = self.eosInv.interpolant.getSinglePoint(rho)
        self.A = integral + np.log(self.targetRho) - 1. + 1./self.N*np.log(2*np.pi*self.N)
        domain = np.in1d(rho,rhonew)
        mufunc = self.A[domain] + self.beta * P/rhonew
        self.mu = data(P, mufunc)

    # plot the chemical potential
    def muPlot(self):
        if isinstance(self.mu,data):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.mu.x,self.mu.y)
            plt.show()
        else:
            print("mu function not calculated")
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The isobar class defines the phase information for
#     integration along an isobar. (See eq. 12 in the chapter)
#-----------------------------------------------------------------------
class isobar(phase):
    def __init__(self, fname, N, P, betaRef, refMu, skipHeader=1):
        # inherit from the phase class (which reads the data)
        phase.__init__(self,fname,N,skipHeader)
        # set parameters
        self.P = P
        self.betaRef = betaRef
        self.refMu = refMu
        self.N = N
        # equation of state
        self.eos = data(self.rawData[:,0],self.rawData[:,1])
        # energy per particle
        self.enp = data(self.rawData[:,0],self.rawData[:,2])
        self.beta = self.eos.x
        self.rho = self.eos.y
        self.enpInfo = self.enp.y
        self.mu = None
        # temperature-dependence
        self.tD = None
        self.params = None
        self.tDcontribution = None

    # set temperature-dependence info
    def tempDependence(self, type, betas=[], params=[]):
        # if the type is tanh, calculate the necessary params
        if type == "tanh":
            self.tD = "tanh"
            assert(len(betas)==4)
            assert(len(params)==2)
            betas.sort()
            idx1 = int(np.argmin(np.abs(self.beta-betas[0])))
            idx2 = int(np.argmin(np.abs(self.beta-betas[1])))
            idx3 = int(np.argmin(np.abs(self.beta-betas[2])))
            idx4 = int(np.argmin(np.abs(self.beta-betas[3])))
            v1 = np.mean(self.enp.y[idx1:idx2+1])
            v2 = np.mean(self.enp.y[idx3:idx4+1])
            print(v1,v2)
            a = (v2-v1)/2
            self.params = [params[0], params[1], a]
        elif type == "WG2008":
            self.tD = "WG2008"
            assert(len(betas)==0)
            assert(len(params)==1)
            self.params = params
        elif type == "S2005":
            self.tD = "S2005"
            assert(len(betas)==0)
            assert(len(params)==1)
            self.params = params


    # calculate enthalpy (U/N + P/rho)
    def calcEnthalpy(self):
        ent = self.enpInfo + self.P / self.rho
        self.enthalpy = data(self.beta, ent)

    # calculate chemical potential (eq. 12)
    def calcMu(self,start=None):
        if start == None:
            start = self.betaRef
        betaF, muF = self.enthalpy.interpolant.integralFunction(start,self.enthalpy.endData,self.betaRef)
        
        # THE FOLLOWING CASES ARE NOT YET ADAPTED TO THE FLUID 
        if self.tD == "tanh":
            tDcontribution = self.params[2]/(self.params[1]*betaF) * 1./np.cosh((1/betaF-self.params[0])/self.params[1])**2
            self.tDcontribution = data(betaF, tDcontribution)
            self.tDcontribution.genInterpolant(method='linear')
            bf, tdf = self.tDcontribution.interpolant.integralFunction(start,self.enthalpy.endData,self.betaRef)
            muF += tdf
        elif self.tD == "WG2008":
            # check if sw is close to u/2
            idx = (np.abs(self.enp.x-self.betaRef)).argmin()
            if np.abs(self.enp.y[idx] - self.params[0]*2) < 2:
                print("sw is very close to u/2")
                # get data range
                idxS = (np.abs(self.enp.x-self.enthalpy.startData)).argmin()
                idxE = (np.abs(self.enp.y-self.enthalpy.endData)).argmin()

                # first fit u0 and sw
                def func(x,u0,sw):
                    return -1*u0 - 2*sw + 2*sw/x
                popt, pcov = curve_fit(func, self.enp.x[idxS:idxE], self.enp.y[idxS:idxE])
                u0 = popt[0]
                sw = popt[1]
                print(u0,sw)

                # calculate integral of (p/rho + u0 -2*sw)
                ent = -1*u0-2*sw + self.P / self.rho
                self.enthalpy2 = data(self.beta, ent)
                self.enthalpy2.genInterpolant(method='linear')
                betaF, muF = self.enthalpy2.interpolant.integralFunction(start,self.enthalpy.endData, self.betaRef)
            else:
                print("sw is fine, proceeding normally")
                tDcontribution = self.params[0]*(-2)/betaF
                self.tDcontribution = data(betaF, tDcontribution)
                self.tDcontribution.genInterpolant(method='linear')
                bf, tdf = self.tDcontribution.interpolant.integralFunction(start,self.enthalpy.endData,self.betaRef)
                muF += tdf
        elif self.tD == "S2005":
            nw = self.params[0]

            Eos = -2
            Eob = -1
            Edb = 1
            Eds = 1.8
            qos = 1
            qob = 10
            qdb = 40
            qds = 49

            deltaEs = Eds - Eos
            deltaEb = Edb - Eob

            def Es(x):
                return (Eos+Eds*np.exp(-x*(Eds-Eos)))/(1+np.exp(-x*(Eds-Eos)))

            def Eb(x):
                return (Eob + Edb*np.exp(-x*(Edb-Eob)))/(1+ np.exp(-x*(Edb-Eob)))
            
            def ew(x):
                return Es(x) - Eb(x)

            def Ss(x):
                return np.log((qos+qds*np.exp(-x*(Eds-Eos)))/(1+np.exp(-x*(Eds-Eos))))
            
            def Sb(x):
                return np.log((qob+qdb*np.exp(-x*(Edb-Eob)))/(1+np.exp(-x*(Edb-Eob))))

            def sw(x):
                return Ss(x) - Sb(x)

            def deps(x):
                return 2*ew(x) - 2/x*sw(x)

            def dSs(x):
                return deltaEs*np.exp(-x*deltaEs)*((1)/(1+np.exp(-x*deltaEs))-(qds)/(qos+qds*np.exp(-x*deltaEs)))
            
            def dSb(x):
                return deltaEb*np.exp(-x*deltaEb)*((1)/(1+np.exp(-x*deltaEb))-(qdb)/(qob+qdb*np.exp(-x*deltaEb)))

            def dEs(x):
                return (deltaEs*np.exp(-x*deltaEs))/(1+np.exp(-x*deltaEs))*(-Eds + (Eos+Eds*np.exp(-x*deltaEs))/(1+np.exp(-x*deltaEs)))

            def dEb(x):
                return (deltaEb*np.exp(-x*deltaEb))/(1+np.exp(-x*deltaEb))*(-Edb + (Eob+Edb*np.exp(-x*deltaEb))/(1+np.exp(-x*deltaEb)))

            def dereps(x):
                return 2*(dEs(x)-dEb(x)) + 2/x**2 * (sw(x)) - 2/x * (dSs(x) - dSb(x))

            tDcontribution = -1*nw*betaF * dereps(betaF)
            self.tDcontribution = data(betaF, tDcontribution)
            self.tDcontribution.genInterpolant(method='linear')
            bf, tdf = self.tDcontribution.interpolant.integralFunction(start,self.enthalpy.endData,self.betaRef)
            muF += tdf

        self.mu = data(betaF, muF + self.refMu )
    
    # plot chemical potential
    def muPlot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.mu.x,self.mu.y)
        plt.show()
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The ideal class defines the phase information for
#     integration along an isotherm. (See eq. 9 in the chapter)
#-----------------------------------------------------------------------
class isotherm(phase):
    def __init__(self, fname, N, rhoRef, beta, refMu, skipHeader=1):
        # initialize phase object
        phase.__init__(self,fname,N,skipHeader)
        # set parameters
        self.rhoRef = rhoRef
        self.beta = beta
        self.refMu = refMu
        # setup equations of state from the input data
        self.eos = data(self.rawData[:,0],self.rawData[:,1])
        self.eosInv = data(self.rawData[:,1], self.rawData[:,0])
        self.mu = None

    # calculate the reference pressure, given the reference rho, using eos
    def calcRefP(self,interp="poly",degree=3):
        self.eosInv.genInterpolant(interp,degree)
        self.Pref = self.eosInv.interpolant.getSinglePoint(self.rhoRef)

    # calculate the integrand of eq. 9
    def calcIntegrand(self):
        integrand = self.beta*self.eos.x / (self.eos.y**2)
        self.integrand = data(self.eos.y,integrand)

    # calculate chemical potential
    def calcMu(self,start=None):
        if start == None:
            start = self.Pref
        rho, mu = self.integrand.interpolant.integralFunction(self.rhoRef,self.integrand.endData,self.rhoRef)
        rhonew, P = self.eosInv.interpolant.getSinglePoint(rho)
        domain = np.in1d(rho,rhonew)
        self.mu = data(P, mu[domain] + self.refMu)
    
    # plot chemical potential
    def muPlot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.mu.x,self.mu.y)
        plt.show()
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#     The ideal class defines the phase information for
#     integration from an Einstein crystal. (See eq. 24 in the chapter)
#-----------------------------------------------------------------------
class einsteinCrystal(phase):
    def __init__(self, fname, gqname, N, rho, targetP, U0, beta, eta, lambdamax, AeoComp,skipHeader=1):
        # create phase object and read Einstein crystal data
        phase.__init__(self,fname,N,skipHeader)
        
        # set parameters
        self.rho = rho
        self.U0 = U0
        self.AeoComp = AeoComp
        self.eta = eta
        self.lambdamax = lambdamax
        self.beta = beta
        self.targetP = targetP

        if gqname !="":
            # read Gaussian Quadrature parameters
            gq = np.genfromtxt(gqname,skip_header=skipHeader)
            self.gq = gq[:,1]
            self.linInterp=0
        else:
            self.linInterp=1
            self.lambdamin = np.amin(self.rawData[:,0])

        # organize data
        self.sq_disp = data(self.rawData[:,0], self.rawData[:,1])
        self.or_disp = data(self.rawData[:,0], self.rawData[:,2])

    # check plateau of the integrand of eq. 19
    # (using eq. 22 and the equation in the appendix of ref. 10)
    def sqDispCheck(self):
        beta = self.beta
        lambdamax = self.lambdamax
        N = self.N
        n = 1.
        a = 1.

        sigma=1.
        # translational component from Frenkel and Ladd, 1984
        sq_disp_Eins_lambda = 1/beta * 3/2 * (N-1)/N * 1/lambdamax
        Pnn_overlap = (erf((beta*lambdamax/2)**0.5*(sigma+a))+erf((beta*lambdamax/2)**0.5 *(sigma-a)))/2 - (np.exp(-beta*lambdamax*(sigma-a)**2/2)-np.exp(-beta*lambdamax*(sigma+a)**2/2))/((2*np.pi*beta*lambdamax)**0.5*a)
        sq_disp_lambda = sq_disp_Eins_lambda - beta * n/2. * 1./(2*a*(2*np.pi*beta*lambdamax)**0.5*(1-Pnn_overlap))*((sigma*a-sigma**2-1/(beta*lambdamax))*np.exp(-beta*lambdamax*(a-sigma)**2/2)+(sigma*a+sigma**2-1/(beta*lambdamax))*np.exp(-beta*lambdamax*(a+sigma)**2/2));
        sq=sq_disp_lambda * N * lambdamax

        # orientational component
        orc=3*N/(2*beta)

        self.plateau = sq + orc

    # calculate integrand of A2 (eq. 19)
    def calcIntegrand(self):
        integrand = self.sq_disp.x * (self.sq_disp.y + self.eta * self.or_disp.y)
        loglambda = np.log(self.sq_disp.x)
        self.integrand = data(loglambda,integrand)

    # plot integrand
    def integrandPlot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(self.integrand.x, self.integrand.y)
        ax.plot([self.integrand.startData, self.integrand.endData],[self.plateau,self.plateau],'r-')
        plt.show()

    # calculate the chemical potential
    def calcMu(self):

        if self.linInterp == 0:
            intGQ = np.sum(self.gq * self.integrand.y)
        else:
            # calculate integral with linear interpolation and the trapezoid rule
            self.integrand.genInterpolant('linear')
            intGQ = self.integrand.interpolant.integrateSingle(np.log(self.lambdamin),np.log(self.lambdamax))

        self.A2 = -self.beta/self.N * intGQ

        self.A1 = self.U0*self.beta/self.N
        self.A3 = 1./self.N * np.log(self.rho)

        self.Aet = -3./2 * (self.N-1)/self.N * np.log(np.pi/(self.beta*self.lambdamax)) - 3./(2*self.N) * np.log(self.N)
        self.Aeo = 3./2*np.log(self.beta*self.lambdamax*self.eta)+self.AeoComp

        self.A = self.A1 + self.A2 + self.A3 + self.Aet + self.Aeo
        self.mu = self.A + self.beta * self.targetP / self.rho
#-----------------------------------------------------------------------
