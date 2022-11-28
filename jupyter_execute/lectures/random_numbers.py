#!/usr/bin/env python
# coding: utf-8

# In[1]:


import holoviews as hv
hv.extension('bokeh')


# In[2]:


import pickle
import numpy as np
import scipy.stats
from IPython.display import YouTubeVideo


# 
# # Simulación y aleatoriedad
# 

# :::{epigraph}
# 
# [What I cannot create, I do not understand](https://jcs.biologists.org/content/joces/130/18/2941.full.pdf)
# 
# -- Richard Feynman
# 
# :::
# 
# 
# La simulación por computadora nos permite modelar sistemas y procesos del mundo real. Si somos capaces de recrear un proceso fielmente entonces habremos entendido sus propiedades. Modelar tiene también muchos usos de gran valor práctico, como por ejemplo utilizar el simulador para predecir el comportamiento futuro del sistema y así tomar decisiones acertadas.
# 
# Sin embargo el mundo real es muy complejo, y por lo general no es posible ni práctico hacer un modelo determinista de los procesos que nos interesan. 
# 
# 
# <a href="https://pijamasurf.com/2013/05/la-fascinante-historia-de-los-28800-mil-patitos-de-hule-que-se-perdieron-en-el-mar/" target="blank"><img src="images/duck.png" width="600"></a>
# 
# 
# Imaginemos por un momento que un barco que lleva patos de hule por el oceano pacífico sufre un accidente en donde su carga se libera en el mar. Si tuvieramos un modelo matemático de como se mueven los patos liberados en el mar podríamos predecir a donde llegarán y luego recogerlos. 
# 
# > **Discusión:** ¿Cómo modelaría el movimiento del pato en el mar? ¿Qué variables habría que considerar? ¿Qué tipo de simplificaciones se podrían hacer?
# 
# 

# Cuando simular de forma determinista se vuelve inmanejable debemos hacer supuestos que simplifiquen el problema. Un supuesto o aproximación bastante usual es
# 
# > Modelo complejo = Modelo simple + Incertidumbre
# 
# Es decir, tratamos lo que es dificil de modelar como un aspecto aleatorio del problema. Más precisamente, lo que hacemos es introducir variables aleatorias en el modelo, es decir pasar de un modelo determinista a un modelo matemático probabilístico. 
# 
# Otra razón para utilizar modelos probabilísticos, es que algunos procesos son inherentemente aleatorios, por ejemplo los fenómenos microscópicos descritos por la mecánica cuántica. 
# 
# > **Discusión:** Algunos autores sostienen que existe aleatoriedad real en algunos procesos [humanos](https://en.wikipedia.org/wiki/Random_walk_hypothesis). ¿Aleatoriedad  real o demasiada complejidad?
# 
# :::{important}
# 
# En este curso nos concentraremos en la simulación de procesos por medio de modelos probabilísticos.
# 
# :::

# ## Números pseudo-aleatorios en Python
# 
# Las variables aleatorias no tienen un valor fijo como sus contrapartes deterministas. El valor que toman está dictado por la distribución que siguen, por ejemplo [Normal, Multinomial, Poisson, entre muchas otras](https://phuijse.github.io/PythonBook/contents/statistics/distributions.html).
# 
# - Muchísimos eventos del mundo real relacionados a promedios de otros eventos se pueden modelar como una distribución normal
# - El lanzamiento de un dado se puede modelar con una distribución multinomial con $k=6$ y $n=1$
# - La cantidad de fotones que golpean el sensor CCD de un telescopio se puede modelar con una distribución de Poisson
# 
# Podemos generar números pseudo-aleatorios que siguen distribuciones específicas en lenguaje Python con la librería [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html). Es necesario leer la documentación de cada distribución para conocer los parámetros que podemos modificar.
# 
# Por ejemplo podemos crear una distribución normal con media 10.0 y desviación estándar 1.0 con:

# In[3]:


dist = scipy.stats.norm(10.0, 1.0)


# Luego podemos generar números pseudo-aleatorios que siguen esta distribución usando el método `rvs`, donde el argumento `size` indica cuantos valores generar:

# In[4]:


sample = dist.rvs(size=1000)

edges, values = np.histogram(sample, bins=10)

hv.Histogram((edges, values)).opts(width=500)


# En general las distribuciones tienen parámetros que debemos ajustar para que nuestro modelo (simulador) sea acorde a la realidad (datos observados). A continuación veremos más en detalle en que consiste un modelo generativo probabilístico.

# ## Ajuste de modelos generativos simples
# 
# Sea un observación descrita por $x$ que viene de una distribución de probabilidad $p^*(x)$. Un modelo generativo es una aproximación 
# 
# $$
# p_\theta(x) \approx p^*(x)
# $$ 
# 
# con la cual podemos generar nuevos ejemplos aleatorios de $x$. Sea por ejemplo la siguiente muestra de diez datos escalares y continuos

# In[5]:


with open("../data/mistery_data.npy", "rb") as f:
    data = np.load(f)
print(data)


# Como se vio en la unidad anterior podemos ajustar un modelo probabilístico usando **estimación de máxima verosimilitud.** Las distribuciones de `scipy.stats` tienen un método llamado `fit` que se encarga de esto:

# In[6]:


# Seleccionamos una distribución
dist = scipy.stats.norm 

# Ajustamos los parámetros con MLE
args = dist.fit(data) 

# Estos son los parámetros ajustados
print(args)

# Luego creamos una distribución con los parámetros ajustados
model = dist(*args) 


# :::{note}
# 
# Los datos fueron creados a partir de una distribución normal con media 10 y desviación estándar 4
# 
# :::
# 
# A pesar de tener pocas observaciones el modelo parece haberse ajustado bien, pero:
# 
# > **Discusión:** ¿Qué pasaría si hubieramos escogido otra distribución para ajustar? ¿Cómo evaluamos si el modelo escogido es adecuado?

# Pasemos ahora a un ejemplo un poco más complejo. Los datos corresponden a imágenes de 28x28 píxeles de dígitos manuscritos del famoso dataset [MNIST](http://yann.lecun.com/exdb/mnist/). Utilizaremos sólo ejemplos del dígito 5
# 
# Ajustemos una distribución normal multivariada y generemos algunos datos

# In[7]:


with open("../data/mnist_data_fives.pkl", "rb") as f:
    data = pickle.load(f)
   
    
dist = scipy.stats.multivariate_normal 
mu = np.mean(data.reshape(-1, 28*28), axis=0) # MLE de la media
cov = np.cov(data.reshape(-1, 28*28), rowvar=False) # MLE de la covarianza
model = dist(mean=mu, cov=cov+0.001*np.eye(28*28)) # Crear modelo
new_data = model.rvs(size=10) # Generar datos


# La fila superior corresponden a ejemplos reales del dataset. La fila inferior corresponden a ejemplos simulados con nuestro modelo generativo "ingenuo". 
# 
# > **Discusión:** ¿Le parecen aceptables los ejemplos generados? ¿Por qué?

# In[8]:


real_list = [hv.Image(sample).opts(width=100, height=100) for sample in data[:10]]
synt_list = [hv.Image(sample.reshape(28, 28)) for sample in new_data]

layout = hv.Layout(real_list + synt_list).cols(10)
layout.opts(hv.opts.Image(width=80, height=80, cmap='binary', xaxis=None, yaxis=None))


# La distribución utilizada anteriormente es demasiado sencilla para este problema. ¿Cómo proponer entonces una distribución más compleja?
# 
# En general es mucho más conveniente asumir que existe una variable oculta o latente $z$ con una distribución sencilla $p(z)$ que luego se modifica a través de una transformación  $p(x|z)$ para obtener un ejemplo de $x$.
# 
# En ese caso buscamos modelar la evidencia o verosimilitud marginal:
# 
# $$
# p(x) = \int_{\mathcal{z\in Z}} p(x|z) p(z) \,dz = \int_{\mathcal{z\in Z}} p(x, z) \,dz
# $$
# 
# Sin embargo calcular esta integral puede ser sumamente costoso. En lecciones futuras veremos como se puede usar el método de Monte Carlo para resolver este problema.

# 
# El siguiente [ejemplo](https://arxiv.org/abs/1807.03039) muestra resultados de un modelo generativo con variable latente entrenado para generar imágenes de rostros humanos.

# In[9]:


YouTubeVideo('exJZOC3ZceA')


# :::{note}
# 
# [Los modelos generativos basados en redes neuronales profundas se investigan muy activamente hoy en día](https://openai.com/blog/generative-models)
# 
# :::

# ## Recordatorio del Teorema de Bayes
# 
# De las propiedades de las probabilidades condicionales y la ley de probabilidades totales podemos escribir
# 
# $$
# p(y|x) = \frac{p(x|y) p(y)}{p(x)} = \frac{p(x|y) p(y)}{\int p(x|y) p(y) \,dy}
# $$
# 
# Tipicamente $x$ representa un conjunto de datos que hemos observado a través de un experimento e $y$ algún parámetro que queremos estimar. 
# 
# **Reflexione:** ¿Qué representan $p(y)$ y $p(y|x)$? ¿Cuándo es conveniente usar el teorema de bayes?
# 
# 
# **Desafio:** La marginalización de la variable latente, el cálculo de la esperanza de una variable continua o el cálculo de la evidencia en el teorema de Bayes requiere resolver integrales que pueden ser muy complicadas
# 
# 

# In[ ]:




