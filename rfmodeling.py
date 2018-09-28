''' Collection of classes for RF active and passive device modeling
__author__ = "Vikram Sekar"
'''

import skrf as rf
import numpy as np

class rfnpn():
    '''Class to create NPN transistor object using scikit RF Network object
    Instance Variables:
        dut = SciKit RF Network object containing raw NPN DUT sparameters
    Methods:
        __init__    - Init method
        d4s         - Kolding 4-step deembedding
        stabfac     - Calculation of Rollett's stability factor
        msg         - Calculation of maximum stable gain in dB
        mag         - Calculation of maximum available gain in dB
        gu           - Calculation of Mason's unilateral gain in dB
        h21         - Calculation of forward current gain
    '''
    def __init__(self, sp):
        ''' Input argument is raw NPN device sparameters '''
        self.data = sp
        self.data.name = sp.name
        self.f = sp.f
        self.z0 = sp.z0
        self.name = sp.name

    def d4s(self, pads='PadShort.s2p', pado='PadOpen.s2p', duto='Open.s2p', shop='ShortOpen.s2p', opsh='OpenShort.s2p'):
        ''' Perform Kolding 4-step deembedding on raw data
        pads = short at the RF pads (simple short)
        pado = open at the RF pads (simple open)
        duto = open at the DUT ref plane (dut open)
        shop = At DUT ref plane, short at P1 and open at P2
        opsh = At DUT ref plane, open at P1 and short at P2

        Returns a new rfnpn object (by returning self)'
        '''

        # Create network objects
        pads = rf.Network(pads)
        pado = rf.Network(pado)
        duto = rf.Network(duto)
        shop = rf.Network(shop)
        opsh = rf.Network(opsh)

        # Simple short Zc
        Zc1 = pads.z[:,0,0]; Zc2 = pads.z[:,1,1]
        Zc = (2/3)* (Zc1.real + Zc2.real)/2  # average contact res of ports, Eq. (6)
        mfactor = np.array( [[3/2, 0], [0, 3/2]] )
        mfactor = np.repeat(mfactor[np.newaxis, :, :], int(np.size(self.f)), axis=0)
        Z1 = np.einsum('ijk,i->ijk', mfactor, Zc)
        self.data.z = self.data.z - Z1 # Eq. (2)

        # Simple open Zp
        Zp1 = pado.z[:,0,0] - pads.z[:,0,0]
        Zp2 = pado.z[:,1,1] - pads.z[:,1,1]
        Zp = (Zp1 + Zp2)/2 # average open impedance of ports, Eq. (7)
        mfactor = np.array( [[1, 0], [0, 1]] )
        mfactor = np.repeat(mfactor[np.newaxis, :, :], int(np.size(self.f)), axis=0)
        Y2 = np.einsum('ijk,i->ijk', mfactor, 1/Zp)
        self.data.y = self.data.y - Y2 # Eq. (3)

        # Extended de-embedding
        # do simple short open deembedding for rest of the fixture structures
        d_duto = rf.Network(); d_duto.z0 = self.z0; d_duto.f = self.f*1e-9
        d_duto.z = duto.z - pads.z; d_duto.y = d_duto.y - pado.y

        d_shop = rf.Network(); d_shop.z0 = self.z0; d_shop.f = self.f*1e-9
        d_shop.z = shop.z - pads.z; d_shop.y = d_shop.y - pado.y

        d_opsh = rf.Network(); d_opsh.z0 = self.z0; d_opsh.f = self.f*1e-9
        d_opsh.z = opsh.z - pads.z; d_opsh.y = d_opsh.y - pado.y

        # Find impedance of dangling grounding leg (DL)
        ZDL = (1/4) * (d_shop.z[:,1,0] + d_shop.z[:,0,1] + d_opsh.z[:,1,0] + d_opsh.z[:,0,1]) # Eq. (15)
        # Find series impedance sum of feed line and ground line
        # but first average the short impedance measurements
        z11s = (1/2) * (d_shop.z[:,0,0] + d_opsh.z[:,1,1])
        alpha = 0 # accuracy adjustment factor (see the Kolding paper)
        ZiZ1 = (z11s - ZDL)/(1+alpha) # Eq. (16)

        a1 = np.dstack((ZiZ1+ZDL, ZDL))
        a2 = np.dstack((ZDL, ZiZ1+ZDL))
        a3 = np.dstack((a1,a2))
        Z3 = np.reshape(a3,(int(np.size(self.f)),2,2))
        self.data.z = self.data.z - Z3 # Eq. (4)

        # Find impedance of the input / output to the dangling gnd leg, and across input/output of dut
        Zio = d_duto.z[:,1,0] + d_duto.z[:,0,0] - 2*ZDL - ZiZ1 # Eq. (17)
        Zf = Zio * ((Zio/(d_duto.z[:,1,0]-ZDL)) - 2) # Eq. (18)

        a1 = np.dstack((1/Zio, -1/Zf))
        a2 = np.dstack((-1/Zf, 1/Zio))
        a3 = np.dstack((a1,a2))
        Y4 = np.reshape(a3,(int(np.size(self.f)),2,2))
        self.data.y = self.data.y - Y4

        return self

    def stabfac(self):
        ''' Calculate Rollett's stability factor k '''
        q1 = np.absolute(self.data.s[:,0,0])**2
        q2 = np.absolute(self.data.s[:,1,1])**2
        q3 = np.absolute(self.data.s[:,0,0]*self.data.s[:,1,1] - self.data.s[:,0,1]*self.data.s[:,1,0])**2
        q4 = np.absolute(self.data.s[:,0,1]*self.data.s[:,1,0])
        k = (1 - q1 -q2 + q3) / (2*q4)
        return np.array(k)

    def msg(self):
        ''' Calculate maximum stable gain (MSG) '''
        msg = np.absolute(self.data.s[:,1,0])/np.absolute(self.data.s[:,0,1])
        msg_dB =[10*np.log10(np.asscalar(x)) for x in msg]
        return np.array(msg_dB)

    def mag(self):
        ''' Calculate maximum available gain (MAG) '''
        k = self.stabfac()
        msg = self.msg()
        mag = (k - np.sqrt(k**2-1))*msg
        mag_dB =[10*np.log10(np.asscalar(x)) for x in mag]
        return np.array(mag_dB)

    def gu(self):
        ''' Calculate the unilateral gain (U) '''
        q1 = self.data.y[:,1,0]
        q2 = self.data.y[:,0,1]
        q3 = self.data.y[:,0,0]; q3 = q3.real
        q4 = self.data.y[:,1,1]; q4 = q4.real
        q5 = self.data.y[:,0,1]; q5 = q5.real
        q6 = self.data.y[:,1,0]; q6 = q6.real
        U = np.absolute(q1-q2)**2 / (4*(q3*q4 - q5*q6))
        gu_dB = [10*np.log10(np.asscalar(x)) for x in U]
        return np.array(gu_dB)

    def h21(self):
        ''' Calculate forward current gain H21 '''
        h21 = self.data.y[:,1,0]/self.data.y[:,0,0]
        return h21

class rfnfet():
    '''Class to create NFET transistor object using scikit RF Network object
    Instance Variables:
        dut = SciKit RF Network object containing raw NPN DUT sparameters
    Methods:
        __init__    - Init method
        d4s         - Kolding 4-step deembedding
        stabfac     - Calculation of Rollett's stability factor
        msg         - Calculation of maximum stable gain in dB
        mag         - Calculation of maximum available gain in dB
        gu           - Calculation of Mason's unilateral gain in dB
        h21         - Calculation of forward current gain
    '''
    def __init__(self, sp):
        ''' Input argument is raw NPN device sparameters '''
        self.data = sp
        self.data.name = sp.name
        self.f = sp.f
        self.z0 = sp.z0
        self.name = sp.name

    def d4s(self, pads='PadShort.s2p', pado='PadOpen.s2p', duto='Open.s2p', shop='ShortOpen.s2p', opsh='OpenShort.s2p'):
        ''' Perform Kolding 4-step deembedding on raw data
        pads = short at the RF pads (simple short)
        pado = open at the RF pads (simple open)
        duto = open at the DUT ref plane (dut open)
        shop = At DUT ref plane, short at P1 and open at P2
        opsh = At DUT ref plane, open at P1 and short at P2

        Returns a new rfnpn object (by returning self)'
        '''

        # Create network objects
        pads = rf.Network(pads)
        pado = rf.Network(pado)
        duto = rf.Network(duto)
        shop = rf.Network(shop)
        opsh = rf.Network(opsh)

        # Simple short Zc
        Zc1 = pads.z[:,0,0]; Zc2 = pads.z[:,1,1]
        Zc = (2/3)* (Zc1.real + Zc2.real)/2  # average contact res of ports, Eq. (6)
        mfactor = np.array( [[3/2, 0], [0, 3/2]] )
        mfactor = np.repeat(mfactor[np.newaxis, :, :], int(np.size(self.f)), axis=0)
        Z1 = np.einsum('ijk,i->ijk', mfactor, Zc)
        self.data.z = self.data.z - Z1 # Eq. (2)

        # Simple open Zp
        Zp1 = pado.z[:,0,0] - pads.z[:,0,0]
        Zp2 = pado.z[:,1,1] - pads.z[:,1,1]
        Zp = (Zp1 + Zp2)/2 # average open impedance of ports, Eq. (7)
        mfactor = np.array( [[1, 0], [0, 1]] )
        mfactor = np.repeat(mfactor[np.newaxis, :, :], int(np.size(self.f)), axis=0)
        Y2 = np.einsum('ijk,i->ijk', mfactor, 1/Zp)
        self.data.y = self.data.y - Y2 # Eq. (3)

        # Extended de-embedding
        # do simple short open deembedding for rest of the fixture structures
        d_duto = rf.Network(); d_duto.z0 = self.z0; d_duto.f = self.f*1e-9
        d_duto.z = duto.z - pads.z; d_duto.y = d_duto.y - pado.y

        d_shop = rf.Network(); d_shop.z0 = self.z0; d_shop.f = self.f*1e-9
        d_shop.z = shop.z - pads.z; d_shop.y = d_shop.y - pado.y

        d_opsh = rf.Network(); d_opsh.z0 = self.z0; d_opsh.f = self.f*1e-9
        d_opsh.z = opsh.z - pads.z; d_opsh.y = d_opsh.y - pado.y

        # Find impedance of dangling grounding leg (DL)
        ZDL = (1/4) * (d_shop.z[:,1,0] + d_shop.z[:,0,1] + d_opsh.z[:,1,0] + d_opsh.z[:,0,1]) # Eq. (15)
        # Find series impedance sum of feed line and ground line
        # but first average the short impedance measurements
        z11s = (1/2) * (d_shop.z[:,0,0] + d_opsh.z[:,1,1])
        alpha = 0 # accuracy adjustment factor (see the Kolding paper)
        ZiZ1 = (z11s - ZDL)/(1+alpha) # Eq. (16)

        a1 = np.dstack((ZiZ1+ZDL, ZDL))
        a2 = np.dstack((ZDL, ZiZ1+ZDL))
        a3 = np.dstack((a1,a2))
        Z3 = np.reshape(a3,(int(np.size(self.f)),2,2))
        self.data.z = self.data.z - Z3 # Eq. (4)

        # Find impedance of the input / output to the dangling gnd leg, and across input/output of dut
        Zio = d_duto.z[:,1,0] + d_duto.z[:,0,0] - 2*ZDL - ZiZ1 # Eq. (17)
        Zf = Zio * ((Zio/(d_duto.z[:,1,0]-ZDL)) - 2) # Eq. (18)

        a1 = np.dstack((1/Zio, -1/Zf))
        a2 = np.dstack((-1/Zf, 1/Zio))
        a3 = np.dstack((a1,a2))
        Y4 = np.reshape(a3,(int(np.size(self.f)),2,2))
        self.data.y = self.data.y - Y4

        return self

    def stabfac(self):
        ''' Calculate Rollett's stability factor k '''
        q1 = np.absolute(self.data.s[:,0,0])**2
        q2 = np.absolute(self.data.s[:,1,1])**2
        q3 = np.absolute(self.data.s[:,0,0]*self.data.s[:,1,1] - self.data.s[:,0,1]*self.data.s[:,1,0])**2
        q4 = np.absolute(self.data.s[:,0,1]*self.data.s[:,1,0])
        k = (1 - q1 -q2 + q3) / (2*q4)
        return np.array(k)

    def msg(self):
        ''' Calculate maximum stable gain (MSG) '''
        msg = self.data.s[:,1,0]/self.data.s[:,0,1]
        msg_dB =[10*np.log10(np.asscalar(x)) for x in msg]
        return np.array(msg_dB)

    def mag(self):
        ''' Calculate maximum available gain (MAG) '''
        k = self.stabfac()
        msg = self.msg()
        mag = (k - np.sqrt(k**2-1))*msg
        mag_dB =[10*np.log10(np.asscalar(x)) for x in mag]
        return np.array(mag_dB)

    def gu(self):
        ''' Calculate the unilateral gain (U) '''
        q1 = self.data.y[:,1,0]
        q2 = self.data.y[:,0,1]
        q3 = self.data.y[:,0,0]; q3 = q3.real
        q4 = self.data.y[:,1,1]; q4 = q4.real
        q5 = self.data.y[:,0,1]; q5 = q5.real
        q6 = self.data.y[:,1,0]; q6 = q6.real
        U = np.absolute(q1-q2)**2 / (4*(q3*q4 - q5*q6))
        gu_dB = [10*np.log10(np.asscalar(x)) for x in U]
        return np.array(gu_dB)

    def h21(self):
        ''' Calculate forward current gain H21 '''
        h21 = self.data.y[:,1,0]/self.data.y[:,0,0]
        return h21
