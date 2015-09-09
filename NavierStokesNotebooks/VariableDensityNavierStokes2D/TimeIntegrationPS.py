import proteus
from proteus.Profiling import logEvent as log

class NonConservativeBackwardEuler(proteus.TimeIntegration.BackwardEuler):
    def __init__(self,transport,integrateInterpolationPoints=False):
        proteus.TimeIntegration.BackwardEuler.__init__(self,transport,integrateInterpolationPoints=integrateInterpolationPoints)

    def calculateElementCoefficients(self,q):
        #for bdf interface
        self.calculateCoefs()
        for ci in self.m_last.keys():
            self.m_tmp[ci][:] = q[('u',ci)]
            q[('mt',ci)][:]   = q[('u',ci)]
            q[('mt',ci)] -= self.m_last[ci]
            q[('mt',ci)] /= self.dt
            q[('mt',ci)] *= q[('dm',ci,ci)]
            for cj in range(self.nc):
                if q.has_key(('dmt',ci,cj)):
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] /= self.dt
                if q.has_key(('dm_sge',ci,cj)) and q.has_key(('dmt_sge',ci,cj)):
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] /= self.dt
            #print q[('mt',ci)]

    def calculateGeneralizedInterpolationCoefficients(self,cip):
        if not self.integrateInterpolationPoints:
            return
        for ci in self.m_ip_last.keys():
            self.m_ip_tmp[ci][:] = cip[('u',ci)]
            cip[('mt',ci)][:]   = cip[('u',ci)]
            cip[('mt',ci)] -= self.m_ip_last[ci]
            cip[('mt',ci)] /= self.dt
            cip[('mt',ci)] *= cip[('dm',ci,ci)]
            for cj in range(self.nc):
                if cip.has_key(('dmt',ci,cj)):
                    cip[('dmt',ci,cj)][:] = cip[('dm',ci,cj)]
                    cip[('dmt',ci,cj)] /= self.dt
                if cip.has_key(('dmt_sge',ci,cj)):
                    cip[('dmt_sge',ci,cj)][:] = cip[('dm_sge',ci,cj)]
                    cip[('dmt_sge',ci,cj)] /= self.dt



class NonConservativeBackwardEuler_cfl(proteus.TimeIntegration.BackwardEuler_cfl):
    def __init__(self,transport,runCFL=0.9,integrateInterpolationPoints=False):
        proteus.TimeIntegration.BackwardEuler_cfl.__init__(self,transport,runCFL,integrateInterpolationPoints=integrateInterpolationPoints)
        #self.runCFL=selfrunCFL
        #self.dtLast=None
        #self.dtRatioMax = 2.0
        #self.cfl = {}
        #for ci in range(self.nc):
        #    if not any(transport.coefficients.cfl):
        #        self.cfl[ci]=numpy.zeros(transport.q[('u',0)].shape,'d')
        #    else:
        #        self.cfl[ci] = transport.coefficients.cfl
        #self.isAdaptive=True

    def calculateElementCoefficients(self,q):
        #for bdf interface
        self.calculateCoefs()
        for ci in self.m_last.keys():
            self.m_tmp[ci][:] = q[('u',ci)]
            q[('mt',ci)][:]   = q[('u',ci)]
            q[('mt',ci)] -= self.m_last[ci]
            q[('mt',ci)] /= self.dt
            q[('mt',ci)] *= q[('dm',ci,ci)]
            for cj in range(self.nc):
                if q.has_key(('dmt',ci,cj)):
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] /= self.dt
                if q.has_key(('dm_sge',ci,cj)) and q.has_key(('dmt_sge',ci,cj)):
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] /= self.dt
            #print q[('mt',ci)]

    def calculateGeneralizedInterpolationCoefficients(self,cip):
        if not self.integrateInterpolationPoints:
            return
        for ci in self.m_ip_last.keys():
            self.m_ip_tmp[ci][:] = cip[('u',ci)]
            cip[('mt',ci)][:]   = cip[('u',ci)]
            cip[('mt',ci)] -= self.m_ip_last[ci]
            cip[('mt',ci)] /= self.dt
            cip[('mt',ci)] *= cip[('dm',ci,ci)]
            for cj in range(self.nc):
                if cip.has_key(('dmt',ci,cj)):
                    cip[('dmt',ci,cj)][:] = cip[('dm',ci,cj)]
                    cip[('dmt',ci,cj)] /= self.dt
                if cip.has_key(('dmt_sge',ci,cj)):
                    cip[('dmt_sge',ci,cj)][:] = cip[('dm_sge',ci,cj)]
                    cip[('dmt_sge',ci,cj)] /= self.dt


class NonConservativeVBDF(proteus.TimeIntegration.VBDF):
    def __init__(self, transport, timeOrder=2, integrateInterpolationPoints=False):
        proteus.TimeIntegration.VBDF.__init__(self, transport, timeOrder, integrateInterpolationPoints)
    def calculateElementCoefficients(self, q):
        self.calculateCoefs()
        log("VBDF calculateElementCoefficients t= %s dt= %s alpha= %s " % (self.t,self.dt,self.alpha_bdf))
        for ci in self.massComponents:
            self.m_tmp[ci][:] = q[('u',ci)]  # use 'u' instead of 'm' to avoid conservativeness
            self.mt_tmp[ci][:]= q[('u',ci)]
            self.mt_tmp[ci] *= self.alpha_bdf
            self.mt_tmp[ci] += self.beta_bdf[ci]
            self.mt_tmp[ci] *= q[('dm', ci, ci)] # multiply by dm/du to get nonconservative coefficient out front
            q[('mt',ci)][:]  = self.mt_tmp[ci]
            for cj in range(self.nc):
                if q.has_key(('dmt',ci,cj)):
                    q[('dmt',ci,cj)][:] = q[('dm',ci,cj)]
                    q[('dmt',ci,cj)] *= self.alpha_bdf
                if q.has_key(('dmt_sge',ci,cj)):
                    q[('dmt_sge',ci,cj)][:] = q[('dm_sge',ci,cj)]
                    q[('dmt_sge',ci,cj)] *= self.alpha_bdf
