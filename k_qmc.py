import numpy as np 

class Occupation(object):
    def __init__(self, k2_max, r_s=1.0):
        filled_k2_max = 0
        self.electrons = []
        # 2D only
        e_shell = int(np.ceil(np.sqrt(k2_max)))
        dim_length = 2*e_shell+1
        self.occupation = np.zeros((dim_length,dim_length,dim_length))
        for i in range(-e_shell,e_shell+1):
            for j in range(-e_shell,e_shell+1):
                for k in range(-e_shell,e_shell+1):
                    if i*i+j*j+k*k <= k2_max:
                        self.occupation[i][j][k] = 1.0
                        self.electrons.append((i,j,k))
                        filled_k2_max = max(filled_k2_max, i*i+j*j+k*k)

        self.N_electrons = np.sum(self.occupation)
        self.rs = r_s
        self.ry_pf = (4.0/np.pi)*0.5 #rydberg prefix

        self.volume = 4.0/3.0*self.N_electrons*np.pi*self.rs**3
        length = self.volume**(1.0/3.0)
        self.invkf2 = 1.0/filled_k2_max
        self.k_fermi = (2*np.pi/length)*(filled_k2_max**0.5)
        self.k_pf = (2.0/np.pi)*(2*np.pi/length)**2
        

    def HF_Energy(self):
        KE = 0
        EE = 0
        for index_1, electron_1 in enumerate(self.electrons):
            KE += self.k_pf*( electron_1[0]**2 + electron_1[1]**2 + electron_1[2]**2 )
            for electron_2 in self.electrons[index_1+1:]:
                delta_k = (electron_1[0]-electron_2[0], electron_1[1]-electron_2[1], electron_1[2]-electron_2[2])
                EE -= 16.0/(self.volume*self.k_pf*( delta_k[0]**2 + delta_k[1]**2 + delta_k[2]**2 ))

        return (KE / self.N_electrons, EE / self.N_electrons, (KE+EE) / self.N_electrons)

    def KE_analytic(self):
        return self.ry_pf*0.6*self.k_fermi**2

    def EE_analytic(self):
        return -self.ry_pf*(2.0*self.k_fermi/np.pi)

    def HF_analytic(self):
        return self.EE_analytic() + self.KE_analytic()


    def HF_infinite(self):
        return 2.2099/(self.rs**2) - 0.916/self.rs

if __name__=='__main__':
    # for r_s in np.arange(4.0,5.0,0.2):
    r_s = 10
    k2_max = 100
    O_9 = Occupation(k2_max,r_s)
    print('{}, {}, {}'.format(r_s, O_9.N_electrons, O_9.HF_Energy()))

