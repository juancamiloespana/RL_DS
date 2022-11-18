# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:25:46 2022

@author: alejandrs
"""
import numpy as np
import random as rn
import requires

class cookies():
    def __init__(self,da_min=1, da_max= 361, dt= 1, verbose=False ): #Verbose imprime la nformación para debugging
        self.dt=dt
        self.da_min=da_min
        self.da_max= da_max
        self.verbose=True
        self.length = self.da_max - self.da_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.da_min, self.da_max, self.dt)
        
    def init_cookies_constants(self,Cap_cedi= 30000, Cap_tras=2000): 
        self.Cap_cedi=Cap_cedi
        self.Cap_tras=Cap_tras
    
    def init_cookies_variables(self ):
        self.Ocup_bodeg_exter= np.full((self.n,), np.nan)
        self.Calc_exces_inv_cedi=np.full((self.n,), np.nan)
        self.A_trasladar=np.full((self.n,), np.nan)
        self.Trans_Nav=np.full((self.n,), np.nan)
        self.Trans_reg=np.full((self.n,), np.nan)
        self.Inv_proy=np.full((self.n,2), np.nan)
        self.Prod_nav=np.full((self.n,), np.nan)
        self.Prod_reg=np.full((self.n,), np.nan)
        self.Demand_prod_nav=np.full((self.n,), np.nan)
        self.Demand_prod_reg=np.full((self.n,), np.nan)
        self.Demand_por_vent=np.full((self.n,2), np.nan)
        self.Cost_total=np.full((self.n,), np.nan)
        self.exces_invent=np.full((self.n,), np.nan)
        
        #Ahora los flujos de entrada 
        self.Cost_diario=np.full((self.n,), np.nan)
        self.Llegada_product=np.full((self.n,2), np.nan)
        self.pena_exceso=np.full((self.n,), np.nan)
        
        #Y con los de salida 
        self.Despach_bodeg=np.full((self.n,2), np.nan)
        self.Despach_desde_cedi=np.full((self.n,2), np.nan)
        
        #Flujo sale de Cedi y entra a Bodeg extern
        self.Exceso_trasla= np.full((self.n,2), np.nan)
        
    def init_cookies_levels(self): #Me falta actualizar niveles daaaa
        self.Cost_bodega_extern= np.full((self.n,), np.nan)
        self.Bodega_externa= np.full((self.n,2), np.nan)
        self.CEDI=np.full((self.n,2), np.nan)
        self.Cost_tot_pena_exces_cedi= np.full((self.n,), np.nan)
      
        
    #Ahora que se iniciaron todas las variables, se procedera con el bucle del tiempo
    
    def run_cookies(self, fast= False):
        if fast:
            self._run_cookies_fast()
        else:
            self._run_cookies()
    
    def _run_cookies(self):
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0(alone=False)
            
        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop",k_)
                self.loopk(k_, alone=False)
    
    def _run_cookies_fast(self):
        self.redo_loop = True
        while self.redo_loop:
            self.redo_loop = False
            self.loop0(alone=False)
            
        for k_ in range(1, self.n):
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop",k_)
                self.loopk(k_, alone=False)
                    
#Funciones para actualizar las variables periodo a periodo
    
   
    def update_Ocup_bodeg_exter(self,i):
        self.Ocup_bodeg_exter[i]=self.Bodega_externa[i].sum()
        print(self.Ocup_bodeg_exter[i])
        
    def update_Calc_exces_inv_cedi(self,i):
        self.Calc_exces_inv_cedi[i]= max((self.Inv_proy[i].sum() - self.Cap_cedi),0)
        print(self.Calc_exces_inv_cedi[i])
        
    def update_A_trasladar(self,i):
        self.A_trasladar[i]=min(self.Calc_exces_inv_cedi[i],self.Cap_tras)
        print(self.A_trasladar[i])
    
    def update_Trans_Nav(self,i):
        self.Trans_Nav[i]=min(self.A_trasladar[i],self.CEDI[i][1])
        print(self.Trans_Nav[i])
    
    def update_Trans_reg(self,i):
        self.Trans_reg[i]=self.A_trasladar[i]-self.Trans_Nav[i]
        
    def update_Prod_nav(self,i):
        self.Prod_nav[i]=rn.randrange(1500,2000,1) #Cambie 0.1 por 1
    
    def update_Prod_reg(self,i):
        self.Prod_reg[i]=rn.randrange(1500,2000,1) #Cambie 0.1 por 1
         
    def update_Demand_prod_nav(self,i):
        self.Demand_prod_nav[i]=rn.randrange(150,3000,1) #Cambie 0.1 por 1
        
    def update_Demand_prod_reg(self,i):
        self.Demand_prod_reg[i]=rn.randrange(1400,1500,2) #Cambie 0.1 por 2
        
    def update_Demand_por_vent(self,i):
        self.Demand_por_vent[i]=np.array([self.Demand_prod_reg[i],self.Demand_prod_nav[i]])
        
    def update_Cost_total(self,i):
        self.Cost_total[i]=self.Cost_tot_pena_exces_cedi[i]+self.Cost_bodega_extern[i]
        
    def update_exces_invent(self,i):
        self.exces_invent[i]=max(self.CEDI[i].sum() - self.Cap_cedi , 0)
        
    def update_Cost_diario(self,i):
        self.Cost_diario[i]=self.Ocup_bodeg_exter[i]*1000
    
    def update_Llegada_product(self,i):
        self.Llegada_product[i]=np.array([self.Prod_reg[i],self.Prod_nav[i]])
        
    def update_pena_exceso(self,i):
        self.pena_exceso[i]=self.exces_invent[i]*3500
        
    def update_Despach_bodeg(self,i):
        self.Despach_bodeg[i][0]=min(self.Bodega_externa[i][0],self.Demand_por_vent[i][0])
        self.Despach_bodeg[i][1]=min(self.Bodega_externa[i][1],self.Demand_por_vent[i][1])
        
    def update_Despach_desde_cedi(self,i):
        self.Despach_desde_cedi[i][0]=min( self.CEDI[i][0],(self.Demand_por_vent[i][0] - self.Despach_bodeg[i][0]))
        self.Despach_desde_cedi[i][1]=min( self.CEDI[i][1],(self.Demand_por_vent[i][1] - self.Despach_bodeg[i][1]))
        
    def update_Inv_proy(self,i):
        self.Inv_proy[i][0]=(self.Llegada_product[i][0]-self.Despach_desde_cedi[i][0])+self.CEDI[i][0]
        self.Inv_proy[i][1]=(self.Llegada_product[i][1]-self.Despach_desde_cedi[i][1])+self.CEDI[i][1]
        
    def update_Exceso_trasla(self,i):
        self.Exceso_trasla[i]=np.array([self.Trans_reg[i]-self.Trans_Nav[i]])
        
    #Ahora vamos a actualizar los niveles 
    #( (entrada - salida )* delta t )+ nivel [ i -1 ]
    
    def update_Cost_bodega_extern(self,i):
        self.Cost_bodega_extern[i] = ( self.Cost_diario[i]*self.dt) + self.Cost_bodega_extern[i-1]
        print(self.Cost_bodega_extern[i])
        
    def update_Bodega_externa(self,i):
        self.Bodega_externa[i]= ( ((self.Exceso_trasla[i] - self.Despach_bodeg[i] ) * self.dt)   +  self.Bodega_externa[i-1])
        print(self.Bodega_externa[i])
        
    def update_CEDI(self,i):
        self.CEDI[i] = ( (self.Llegada_product[i] - self.Despach_desde_cedi[i] )*self.dt ) + self.CEDI[i-1]
        print(self.CEDI[i])
        
    def update_Cost_tot_pena_exces_cedi(self,i):
        self.Cost_tot_pena_exces_cedi[i]= ( (self.pena_exceso[i] * self.dt ) + self.Cost_tot_pena_exces_cedi[i-1])
        
        
    def loop0(self,alone=False):
        
        #Los niveles deben iniciarse en 0 
        
        self.Cost_bodega_extern[0]=0
        self.Bodega_externa[0]=0
        self.CEDI[0] = 0
        self.Cost_tot_pena_exces_cedi[0] = 0
        
        #Ahora el resto
        
        self.update_Prod_reg(0)
        self.update_Prod_nav(0)
        self.update_Llegada_product(0)
        self.update_Demand_prod_nav(0)
        self.update_Demand_prod_reg(0)
        self.update_Demand_por_vent(0)
        self.update_Despach_bodeg(0)
        self.update_Despach_desde_cedi(0)
        self.update_Inv_proy(0)
        self.update_Calc_exces_inv_cedi(0)
        self.update_A_trasladar(0)
        self.update_Trans_Nav(0)
        self.update_Trans_reg(0)
        self.update_Exceso_trasla(0)
        self.update_Ocup_bodeg_exter(0)
        self.update_Cost_diario(0)
        self.update_Cost_total(0)
        self.update_exces_invent(0)
        self.update_pena_exceso(0)
        
        
    def loopk(self,i, alone=False):
        
        #Los niveles deben iniciarse en 0 
        
        self.update_Cost_bodega_extern(i)
        self.update_Bodega_externa(i)
        self.update_CEDI(i)
        self.update_Cost_tot_pena_exces_cedi(i)
        
        #Ahora el resto
        
        self.update_Prod_reg(i)
        self.update_Prod_nav(i)
        self.update_Llegada_product(i)
        self.update_Demand_prod_nav(i)
        self.update_Demand_prod_reg(i)
        self.update_Demand_por_vent(i)
        self.update_Despach_bodeg(i)
        self.update_Despach_desde_cedi(i)
        self.update_Inv_proy(i)
        self.update_Calc_exces_inv_cedi(i)
        self.update_A_trasladar(i)
        self.update_Trans_Nav(i)
        self.update_Trans_reg(i)
        self.update_Exceso_trasla(i)
        self.update_Ocup_bodeg_exter(i)
        self.update_Cost_diario(i)
        self.update_Cost_total(i)
        self.update_exces_invent(i)
        self.update_pena_exceso(i)
        
        
    
     
        
def hello_cookies():
    
    Cookies = cookies()
    Cookies.init_cookies_constants()
    Cookies.init_cookies_variables()
    Cookies.init_cookies_levels()
    Cookies.run_cookies(fast=True)
    print(Cookies.Cost_total)

if __name__ == "__main__":
    hello_cookies()
    