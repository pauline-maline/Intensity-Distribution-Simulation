# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import fresnel
from scipy.ndimage.filters import gaussian_filter


# specify the following values of your radar setup:
antennatilt = 15            # angle between antenna alignment and the horizon [°].
hQ = 1.985                  # height of the radar antenna aboove ground [m].
a = 6                       # distance between the radar and the clutter shielding fence [m].
hZ = 2.2                    # height of the clutter shielding fence [m].
lamda = 0.031859            # radar wavelength [m].
Pt = 12000                  # peak transmit power of the radar [W].
Gt = 1247.38                # antenna gain of the radar [-].

# specify the effective area of the radar antenna [m²]
AEr= (Gt*lamda**2)/(4*math.pi)


# specify the following values of your measurement object:
sigma = 0.00127      # radar cross section [m²].


# specify parameters for the plot of the  intensity distribution:
x_dezimeter = 0      # scale of the x-axis: 1 for [dm], 0 for [m].
y_dezimeter = 0      # scale of the x-axis: 1 for [dm], 0 for [m].
x_resolution = 1000  # integer value of maximum distance [m]
y_resolution = 300   # integer value of maximum height [m]
tr = [-74]           # threshold of the minimum return power for object detection by the radar [dBm].
sigma1 = 0.7         # value of sigma for Gaussian filter. 


# end of spefying parameters.
######################################

# automatic selection of plot configurations:
if y_dezimeter == 1:
    y_faktor = 10
    y_unit = 'Height above ground [dm]'
else:
    y_faktor = 1
    y_unit = 'Height above ground [m]'
    
if x_dezimeter == 1:
    x_faktor = 10
    x_unit = 'Horizontal distance [dm]'
else:
    x_faktor = 1
    x_unit = 'Horizontal distance [m]'
    



# calculation of the angle between the straight lines connecting the radar antenna to fence edge and the radar antenna to horizon [°]:
alpha = math.atan((hZ-hQ)/a)*360/(2*math.pi)-antennatilt

# calculation of the angle-dependent vertical gain factor of the radar antenna on edge (fK; should be betweeen 0 and 1) [-]:
if alpha >=-90 and alpha <= -66: #alpha between -90° and -66°
    fK = 10**((-0.0000014949373086659*alpha**6-0.00070852557198875*alpha**5-0.139564445915408*alpha**4-14.6269997026561*alpha**3-860.348932178637*alpha**2-26930.8173654556*alpha-350531.287449198)/10)
elif alpha < -65:  #alpha between -66° and -65°
    fK = 10**((-25)/10)
elif alpha <= -46: #alpha between -65° and -46°
    fK = 10**((0.00000924361736032164*alpha**6+0.00317711838901613*alpha**5+0.452725277285858*alpha**4+34.2289384878583*alpha**3+1448.03189137727*alpha**2+32495.1484198473*alpha+302155.154934733)/10)
elif alpha < -45:  #alpha between -46° and -45°
    fK = 10**((-28.5)/10)
elif alpha <= -6: #alpha between -45° and -6°
    fK = 10**((0.00000023872326298364*alpha**6+0.0000323677929426722*alpha**5+0.00162989131166868*alpha**4+0.0378547726457397*alpha**3+0.410376348505567*alpha**2+2.40825420520235*alpha+5.54984371908995)/10)
elif alpha < -5:    #alpha between -6° and -5°
    fK = 10**((-0.3)/10)
elif alpha <= 24: #alpha between -5° and 24°
    fK = 10**((0.00000032995287249996*alpha**6-0.0000163823803811039*alpha**5+0.000265198637203808*alpha**4-0.00142243569562971*alpha**3-0.0225810237465976*alpha**2+0.00733935885191899*alpha+0.09174627946489840000)/10)
elif alpha < 25:  #alpha between 24° - 25°
    fK = 10**((-12)/10)
elif alpha < 48: #alpha between 25° - 48°
    fK = 10**((0.00000077622007460532*alpha**6-0.000220951871694131*alpha**5+0.0239971036813745*alpha**4-1.31399907413014*alpha**3+38.829659677977*alpha**2-592.007132559122*alpha+3644.36331641867)/10)
elif alpha < 49: #alpha between 48° - 49°
    fK = 10**((-45)/10)
elif alpha < 90: #alpha between 49° - 90° 
    fK = 10**((-0.0000000561726448663*alpha**6+0.0000249198038587816*alpha**5-0.00464433823247045*alpha**4+0.465663117836564*alpha**3-26.4723013262135*alpha**2+807.350584298887*alpha-10311.7349777897)/10)


# declaration of arrays for calculations:
Bf   = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double) 
Ibe  = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double) 
Iad  = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double) 
Igr  = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double) 
Iges = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double) 
Pe   = np.ones((y_resolution*y_faktor, x_resolution*x_faktor), dtype=np.double)
 

# calculation of diffraction:
for iy in range (0, y_resolution*y_faktor):
    for ix in range (1, (x_resolution*x_faktor+1)):
            y1=y_resolution*y_faktor-1-iy
            x1=ix-1
            
            
            d = a*(iy/y_faktor-hQ)/(ix/x_faktor+a)   # [m].    
            y = hZ - hQ - d       # [m].
            roh0 = math.sqrt(a**2 + d**2) # [m].
            r0 = math.sqrt((ix/x_faktor)**2 + (iy/y_faktor-hQ-d)**2) # [m].
            u = y*math.sqrt(2*(roh0+r0)/(lamda*roh0*r0))
            B = fresnel(u)            # Fresnel integrals.
            r = math.sqrt((a+ix/x_faktor)**2 + (iy/y_faktor-hQ)**2) # distance from radar antenna to viewing point (direct way) [m].         
            Bf[y1][x1]  = 1/2*((0.5-B[1])**2 + (0.5-B[0])**2) # diffraction factor. 
            
             
            if 0.985 <= Bf[y1][x1] <= 1.015 and ix < 200:
               Bf[y1][x1] = 1
            
            Ibe[y1][x1] = Bf[y1][x1]**2 *Pt *Gt *sigma *fK**2 /(16*math.pi**2*r**4)

       
            # calculation of the angle between the straight line from the radar antenna viewing point and the main beam direction of the radar antenna [°]:
            phi = math.atan((iy/y_faktor-hQ)/(a+ix/x_faktor))*360/(2*math.pi) - antennatilt 
            
            if phi <= alpha: 
                fP = fK 
            elif phi >=-90 and phi <= -66: #phi between -90° and -66°
                fP = 10**((-0.0000014949373086659*phi**6-0.00070852557198875*phi**5-0.139564445915408*phi**4-14.6269997026561*phi**3-860.348932178637*phi**2-26930.8173654556*phi-350531.287449198)/10)
            elif phi < -65:  #phi between -66° and -65°
                fP = 10**((-25)/10)
            elif phi <= -46: #phi between -65° and -46°
                fP = 10**((0.00000924361736032164*phi**6+0.00317711838901613*phi**5+0.452725277285858*phi**4+34.2289384878583*phi**3+1448.03189137727*phi**2+32495.1484198473*phi+302155.154934733)/10)
            elif phi < -45:  #phi between -46° and -45°
                fP = 10**((-28.5)/10)
            elif phi <= -6: #phi between -45° and -6°
                fP = 10**((0.00000023872326298364*phi**6+0.0000323677929426722*phi**5+0.00162989131166868*phi**4+0.0378547726457397*phi**3+0.410376348505567*phi**2+2.40825420520235*phi+5.54984371908995)/10)
            elif phi < -5:    #phi between -6° and -5°
                fP = 10**((-0.3)/10)
            elif phi <= 24: #phi between -5° and 24°
                fP = 10**((0.00000032995287249996*phi**6-0.0000163823803811039*phi**5+0.000265198637203808*phi**4-0.00142243569562971*phi**3-0.0225810237465976*phi**2+0.00733935885191899*phi+0.09174627946489840000)/10)
            elif phi < 25:  #phi between 24° - 25°
                fP = 10**((-12)/10)
            elif phi < 48: #phi between 25° - 48°
                fP = 10**((0.00000077622007460532*phi**6-0.000220951871694131*phi**5+0.0239971036813745*phi**4-1.31399907413014*phi**3+38.829659677977*phi**2-592.007132559122*phi+3644.36331641867)/10)
            elif phi < 49: #phi between 48° - 49°
                fP = 10**((-45)/10)
            elif phi < 90: #phi between 49° - 90° 
                fP = 10**((-0.0000000561726448663*phi**6+0.0000249198038587816*phi**5-0.00464433823247045*phi**4+0.465663117836564*phi**3-26.4723013262135*phi**2+807.350584298887*phi-10311.7349777897)/10)
   
                        
            Iad [y1][x1] = Pt *Gt *sigma *fP**2/(16*math.pi**2*r**4)
            Igr [y1][x1] = Pt *Gt *sigma *fK**2/(16*math.pi**2*r**4)                     
            Iges [y1][x1] = (math.sqrt(Iad[y1][x1]) + math.sqrt(Ibe[y1][x1]) - math.sqrt(Igr[y1][x1]))**2          
            Pe  [y1][x1] = AEr * Iges [y1][x1]            
            if Pe [y1][x1] >0:
                Pe  [y1][x1] = math.log10((Pe[y1][x1])*1000) * 10            
            if Iges[y1][x1] > 0:
                Iges [y1][x1] = math.log10((Iges[y1][x1])*1000) * 10            
            if Iad[y1][x1] > 0:
                Iad  [y1][x1] = math.log10((Iad[y1][x1])*1000) * 10                
            if Ibe[y1][x1] > 0:
                Ibe  [y1][x1] = math.log10((Ibe[y1][x1])*1000) * 10
            if Igr[y1][x1] > 0:
                Igr  [y1][x1] = math.log10((Igr[y1][x1])*1000) * 10

            


# Plot the resulting intensity distribution: Diffraction representation of the intensity distribution.
# "Pe" = diffraction representation of the intensity distribution [dBm].      
fig=plt.figure(figsize=(16, 5))
levels = np.linspace(-160, 0, 120)
xlist = np.linspace(0, (x_resolution*x_faktor-1), x_resolution*x_faktor)
ylist = np.linspace(y_resolution*y_faktor, 0, y_resolution*y_faktor)
X, Y = np.meshgrid(xlist, ylist)
cs = plt.contourf(X, Y, gaussian_filter(Pe,sigma1), levels=levels)
cs2 = plt.contour(X,Y,gaussian_filter(Pe,sigma1),tr,colors='r',linestyles='solid')
plt.clabel(cs2)
plt.colorbar(cs, ticks=[0, -20, -40, -60, -80, -100, -120, -140], label='Power [dBm]')
plt.title('Power distribution with a clutter shielding fence and detection threshold of '+str(tr[0])+' dBm')
plt.xlabel(x_unit)
plt.ylabel(y_unit)
plt.show()