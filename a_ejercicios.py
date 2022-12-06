
import pandas as pd
from random import seed
from random import random as rm
from random import sample
import numpy as np
import plotly.express as px
import nbformat



def k_armed_bandit(episodes=1000,k=10,ep=0.1,s_seed=1234):


    #### initializar variables
    seed(s_seed) ###establecer semilla    
    df=pd.DataFrame()

    df['A'] = list(range(1,k+1))
    df['Q'] =0 ### Inicializar acumulado de valor de la acción
    df['N']= 0 ### Iniicalizar acumulado de número de veces de la acción
    df["mean_value"]=sample(list(range(50)),k) ##asignar un valor de media para cada brazo
    df["value_var"]=sample(list(range(1,15)),k) ##asignar un valor de varianza para cada brazo
    
    
    for i in range(episodes):
        
    
        ###Generar número aleatorio para exploración/explotación
        seed(i)
        rnd=rm()
        if(rnd >= ep):
            seed(i)
            max=df['Q'].max()
            bestAs=df['A'][df['Q']==max]
            a=bestAs.sample(1).values[0]
            a_index=df[df['A']==a].index.values[0] ## obtener indice de fila
            
        else:
            seed(i)
            a=df['A'].sample().values[0]
            a_index=df[df['A']==a].index.values[0] ## obtener indice de fila
            
        ###Sacar informcaión de acción seleccionada
        seed(i)
        mu=df[df['A']==a]['mean_value'].values[0]
        sig=df[df['A']==a]['value_var'].values[0]
        R=np.random.normal(mu,sig,1)[0]

        #### Generar valores de N Q para acción seleccionada

        N_A=df.loc[a_index,'N']+1 ## actualizar N
        Q_A = df.loc[a_index,'Q'] ### Q acumulado hasta t-1
        Q_A=Q_A+((R-Q_A)/N_A)  ## Q actualizado t

        ## guardar valores de N y Q en tabla
        df.loc[a_index,'N'] =  N_A 
        df.loc[a_index,'Q'] = Q_A
        
        
    return(df)






df=k_armed_bandit()
fig=px.bar(data_frame=df, y='N',x='A', color="Q", title="Frecuencia de acciones y su valores")
fig.show()

fig=px.bar(data_frame=df, y='Q',x='A', color="N", title="Frecuencia de acciones y su valores")
fig.show()

#final













