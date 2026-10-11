# Supuestos de la optimización (ronda 1)

> **Advertencia honesta:** el enunciado no trae costos, márgenes ni capacidades. Todas las
> cifras de abajo son **supuestos del equipo**, razonados desde la lógica de la banca retail
> peruana, **sin una fuente publicada que las respalde**. Por eso cada una tiene un rango y
> el tornado (`reports/sens_tornado.csv`, `reports/fig/opt_tornado.png`) muestra cuánto cambia
> la decisión si el valor real está en el extremo. Viven en `src/config.py::OPT` (valor
> central) y `src/config.py::OPT_RANGOS` (rango). El día del reto se reemplazan por las cifras
> oficiales, si las hay.

## Valor por cliente

`valor_i = clip(tasa_margen × ingresos, valor_min, valor_max) × (1 − pérdida_esperada[banda]) × incrementalidad`

| Supuesto | Central | Rango | Por qué |
|---|---|---|---|
| `tasa_margen` | 1,5 % del ingreso anual | 1,0 % – 2,0 % | Margen neto que deja un producto nuevo (tarjeta, seguro, depósito o préstamo pequeño) durante su vida útil. Con un ingreso medio de ~S/ 65 000 da ~S/ 975 antes de pérdida: del orden de un margen anual de una tarjeta o un préstamo de consumo pequeño. Antes era 2 %, que nos pareció optimista. |
| `valor_min` / `valor_max` | S/ 100 / S/ 1 500 | — | Piso: hasta el cliente de menor ingreso deja algo (comisiones). Techo: un solo producto de campaña no escala sin límite con el ingreso. Antes el techo era S/ 2 500. |
| `perdida_esperada` | low 2 %, medium 6 %, high 15 % | high 10 % – 25 % | Pérdida esperada (≈ PD × LGD) que se descuenta del margen. Crece con la banda de riesgo: es el castigo económico al riesgo, aparte del tope de la C5. |
| `incrementalidad` | 50 % | 30 % – 70 % | **Supuesto nuevo y el más importante para la credibilidad.** El modelo predice quién convierte, no quién convierte *gracias al contacto*. Una parte de los que convierten lo habría hecho igual. Sin grupo de control no se puede medir, así que atribuimos solo la mitad del margen al contacto y proponemos un piloto A/B para medirla. |

## Canales

`EV[i,c] = p_i × efectividad_c × valor_i − costo_c`. La probabilidad `p_i` es la del modelo
**calibrada** (isotónica, ajustada solo con meses anteriores al mes evaluado) y se interpreta
como la propensión con el canal de referencia (call center, efectividad 1).

| Canal | Costo por contacto (rango) | Efectividad (rango) | Cupo mensual (rango) | Por qué |
|---|---|---|---|---|
| App / push | S/ 0,30 (0,10 – 1,00) | 0,50 (0,30 – 0,70) | sin tope | Costo marginal casi nulo (notificación, SMS o banner personalizado); el rango alto incluye SMS y diseño de la pieza. Convierte la mitad que una llamada porque no hay persuasión. Solo para clientes con `activo_movil` (C4). |
| Call center | S/ 9 (6 – 15) | 1,00 (referencia) | 1 500 (750 – 2 000) | Un agente cuesta del orden de S/ 2 500 a 3 500 al mes con cargas y logra pocas decenas de contactos efectivos al día; dividido entre contactos efectivos, da un costo de un dígito a poco más de 10 soles. El cupo de 1 500 equivale a un equipo pequeño dedicado parcialmente a la campaña. |
| Asesor en agencia | S/ 35 (25 – 60) | 1,50 (1,2 – 2,0) | 400 (200 – 600) | Tiempo del asesor (30 a 45 minutos con cita) y el costo de oportunidad del puesto. Cara a cara convierte más. Solo si la sucursal está a ≤ 10 km (C4). |

## Reglas de la campaña

| Supuesto | Central | Rango | Por qué |
|---|---|---|---|
| `presupuesto` | S/ 25 000 al mes | S/ 15 000 – 40 000 | ~S/ 2,5 por cliente del mes. Lo fija el área comercial; la curva presupuesto → valor dice cuánto conviene. |
| `max_share_riesgo_alto` (C5) | 10 % de los contactados | 5 % – 20 % | Apetito de riesgo: el 21 % de la base está en banda `high`; limitarlo a la mitad de su peso es una política prudente. Es una decisión de riesgos, no comercial. |
| `min_dias_desde_interaccion` (C6) | 7 días | 3 – 30 | No saturar al cliente que acaba de tener contacto con el banco (experiencia y conducta de mercado). |
| `asesor_max_km` (C4) | 10 km | — | El asesor solo tiene sentido si el cliente puede ir a la agencia. |
| Un contacto por cliente (C1) | — | — | Fatiga del cliente: una oferta al mes por un solo canal (supuesto S2 de `RETO.md`). |
| Solo EV > 0 (C7) | — | — | No gastar en contactos que destruyen valor. |

## Qué no sabemos (y cómo se cerraría)
1. **Incrementalidad real:** piloto A/B con un grupo de control al azar del 10 % en la primera campaña.
2. **Efectividad por canal:** el mismo piloto, aleatorizando el canal dentro de un segmento.
3. **Costos y cupos reales:** los entrega el área comercial; el motor no cambia, solo `config.OPT`.
4. **Calibración en diciembre:** diciembre (gratificación) puede tener otra tasa base. Las
   probabilidades se calibran con meses anteriores; si la tasa sube, el valor esperado del
   reporte queda **conservador**.
