## 8. ¿Qué significa para el banco?

Cifras del mes de validación t=10 (octubre), con el modelo `ens_3` y los supuestos centrales de
[`supuestos_opt.md`](supuestos_opt.md). Todas salen de `reports/optimizacion_estrategias.csv` y
de los `reports/sens_*.csv`.

1. **Con los mismos S/ 25 000, el óptimo rinde 3,4 veces lo que rinde contactar al azar:**
   S/ 389 mil de valor esperado neto contra S/ 114 mil (S/ 112 mil ± 2 mil en 20 semillas).
   Con las conversiones que sí ocurrieron ese mes, los contactados por el óptimo convirtieron
   **1 275 veces contra 277**: el costo por venta baja de **~S/ 90 a ~S/ 20**.
2. **El modelo vale ~S/ 94 mil por campaña** (greedy por probabilidad − aleatorio) y **la
   optimización suma ~S/ 181 mil más** (óptimo − greedy por probabilidad). Siendo honestos:
   frente a un analista que ya ordena por rentabilidad (valor/costo), el solver agrega
   S/ 34 mil (+9,6 %). El gran salto es **elegir el canal correcto**: app barata para la
   propensión media (5 238 clientes), call center al tope (1 499) y asesor solo para alto
   valor (283 de 400 cupos).
3. **Precios sombra (dónde poner el próximo sol):**
   - **S/ 1 000 más de presupuesto ≈ S/ 2 900 a 3 000 más de valor**, pero solo hasta
     ~S/ 29 000. Desde ahí el valor se congela en S/ 401 mil: ya se contactó a todo cliente
     con valor esperado positivo y el límite pasa a ser la capacidad.
   - **Un cupo más de call center vale ~S/ 33** netos (32,1 a 32,9 según el método): más
     agentes en campaña conviene mientras cuesten menos que eso por contacto adicional.
   - **Un punto porcentual más de apetito de riesgo alto vale ~S/ 1 100**: la política de
     riesgo cuesta poco, y la recomendación es **no relajarla**.
   - **Un cupo más de asesor vale S/ 0**: hoy sobran 117 cupos de asesor. Esas horas se
     pueden usar en otra campaña.
4. **Escenarios:** si el call center pierde la mitad de sus cupos, el valor cae 10 %
   (S/ 389 mil → S/ 351 mil) y el óptimo lo compensa con más app. Endurecer el riesgo alto de
   10 % a 5 % cuesta solo 1,8 % (S/ 7 mil). Un asesor 50 % más caro cuesta 3,5 % (S/ 14 mil).
   En los cuatro escenarios el óptimo le gana a los otros tres métodos. (Con menos cupos de call
   center el aleatorio y el greedy por probabilidad *suben*: al no poder gastar en llamadas,
   gastan en app. Confirma que su problema es el canal, no el orden.)
5. **Lo que más mueve el resultado no es un costo sino la incrementalidad** (qué parte de la
   venta se debe al contacto). Entre 30 % y 70 %, el valor va de S/ 223 mil a S/ 555 mil. Le
   siguen el margen por producto (S/ 254 a 481 mil) y la efectividad de la app (S/ 322 a 464 mil).
   La decisión de **a quién** contactar es estable; lo incierto es **cuánto** vale. Por eso
   proponemos un **piloto A/B** con 10 % de control en la primera campaña.
6. **No depende de una calibración frágil:** isotónica y Platt dan el mismo Brier (0,1273), un
   valor que difiere en 1 % y **95 % de los mismos clientes elegidos**.

**Proyección anual (estimación, con los supuestos centrales):** ~S/ 389 mil × 12 ≈ **S/ 4,7
millones** de valor esperado neto, de los cuales ~S/ 3,3 millones (S/ 275 mil × 12) son
incrementales frente a contactar al azar. Es una cifra de orden de magnitud, no un compromiso:
se valida con el piloto.
