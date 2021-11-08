#Intento despejar A y B en el equilibrio
#%%
#El código en esta celda asegura que SymPy funcione correctamente en Google colab o en sus computadoras
import os
if "COLAB_GPU" in os.environ:
    from sympy import *
    def custom_latex_printer(expr, **options):
        from IPython.display import Math, HTML
        from google.colab.output._publish import javascript
        url = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_CHTML"
        javascript(content="""window.MathJax = { tex2jax: { inlineMath: [ ['$','$'] ], processEscapes: true } };""")
        javascript(url=url)
        return latex(expr, **options)
    init_printing(use_latex="mathjax", latex_printer=custom_latex_printer)
else:
    from sympy import *
    init_printing()

#%%
#Definimos las constantes
S = symbols('S', real=True) #El estímulo
K_SA, K_SB = symbols('K_SA, K_SB', real=True) #Acoplamiento del estímulo
K_BA, K_AB = symbols('K_BA, K_AB', real=True) #Influencias de A y B entre ellos
A, B = symbols('A, B', real=True) #A y B

#%%
#Resuelvo el sistema
from sympy.solvers import solve

solve([(S*K_SA*(1-A))/(K_SA+1-A)-(K_BA*A*B)/(K_BA+A), (S*K_SB*(1-B))/(K_SB+1-B)-(K_AB*A*B)/(K_AB+B)], (A, B), simplify=True, force=True)
