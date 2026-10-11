# EDA — `train.csv` (110.100 × 25). No se ha leído `test.csv`.

> **Estado de este documento (03/10, 15:41).** Etapa 2 de 4. El perfil del panel y
> la auditoría de fuga siguen siendo válidos: `meses_restantes` con AUC 1,0000 es
> la fuga que justifica la lista de variables (`artifacts/features.md`) y el
> `features.md` se apoya en ella.
>
> Lo único acotado en el tiempo: cuando se escribió esto, `test.csv` no se había
> leído. **La etapa 4 sí lo leyó**, y lo hizo al final, con las variables ya
> cerradas. Lo que se midió de diciembre está en `artifacts/modelo.md` §11 (PSI) y
> en `artifacts/resultado_final.md`.

## 1. Integridad: todo cuadra, incluidos los facts de `AGENTS.md`

| Comprobación | Resultado |
|---|---|
| Filas × columnas | 110.100 × 25 |
| `metaData.csv` | 26 filas; la 26 (`prediccion`) **no está en train** — es la de entrega |
| Nulos | **0** en 2.752.500 celdas |
| `inf` | 0 |
| Duplicados en `(id_cliente, mes)` | **0** → la clave compuesta es única |
| Clientes únicos | 24.628 |
| `id_cliente` | 1 → 24.628, sin huecos: es un **índice, no un ID aleatorio** |
| Tasa de `objetivo` | **15,0472 %** (16.567 positivos) |
| Tasa por mes | 14,071 – 15,745 % → amplitud **1,674 pp** (plana) |
| Filas por mes | 9.500 – 10.400 (no constante) |

**Tipo real vs declarado**: 21/25 coinciden. `objetivo` llega como `int64` aunque `metaData` dice `boolean`. Las 5 categóricas string llegan como `object`, las 5 banderas como `bool` de verdad.

## 2. La regla de truncamiento: verificada, 0 violaciones

| Test | Resultado |
|---|---|
| Clientes con `objetivo=1` | 16.567 |
| Clientes con `objetivo=1` en **más de un mes** | **0** |
| Filas `objetivo=1` en un mes **posterior** al primer positivo del cliente | **0** |
| Filas `objetivo=0` **después** de su mes positivo | **0** |

El dataset **cumple** la regla de `metaData.csv`. Y trae una consecuencia que `AGENTS.md` no dice:

> **16.567 de 24.628 clientes convierten alguna vez = 67,3 %.** Las 8.061 filas que nunca convierten son el **44,5 % de todas las filas** y aportan **0 positivos**.
> Tasa dentro de convertibles: **27,11 %**. Dentro de no convertibles: **0,00 %**.

El 15 % por fila es una **mezcla de dos poblaciones**, no una tasa de riesgo homogénea. Eso manda sobre cómo se plantea el modelo.

## 3. Hallazgo que reencuadra todo el feature engineering

**20 de las 21 predictoras son constantes al 100 % dentro del cliente.**

| Variable | valores por cliente (medio) | % clientes que nunca cambian |
|---|---|---|
| `edad`, `ingresos`, `ratio_deuda_ingresos`, `antiguedad_cuenta_meses`, `numero_productos`, `saldo_promedio`, `dias_ultima_transaccion`, `antiguedad_direccion_meses`, `visitas_web_ultimos_90_dias`, `distancia_sucursal_km`, `dia_preferido_pago` | 1,000 | **100,00 %** |
| `ocupacion`, `region`, `canal_adquisicion`, `banda_riesgo`, `dispositivo_principal` | 1,000 | **100,00 %** |
| `tiene_tarjeta_credito`, `activo_movil`, `es_nuevo_cliente`, `tiene_prestamo`, `tiene_seguro` | 1,000 | **100,00 %** |
| **`dias_ultima_interaccion`** | **2,822** | **34,42 %** ← la única que se mueve |

Consecuencia directa y dura: **no hay deltas, ni medias móviles, ni volatilidades, ni pendientes que construir.** El 81 % del trabajo habitual de un panel no existe aquí. El panel es una ficha de cliente repetida mes a mes hasta que convierte.

### Y la única variable que se mueve es ruido

`dias_ultima_interaccion` es `dias_ultima_transaccion` en el **54,93 %** de las filas; cuando difiere, el valor es un **redraw i.i.d. de la misma distribución** (media 191,7 vs 188,9; desv 104,7 vs 105,0). Ejemplos reales de un mismo cliente:

| mes | `dias_ultima_transaccion` (fijo) | `dias_ultima_interaccion` |
|---|---|---|
| 202606 | 304 | 304 |
| 202607 | 304 | 137 |
| 202608 | 304 | 70 |
| 202609 | 304 | 302 |
| 202610 | 304 | 279 |

Cero tendencia, cero persistencia. **No hay dinámica de comportamiento que capturar.** Los "agregados" que salen de aquí son proxies del número de meses, no del comportamiento del cliente.

## 4. Tabla de variables (resumen)

| variable | declarada | real | nulos | distintos | dominante (conc %) | min | p01 | p25 | mediana | media | p75 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `id_cliente` | int | int64 | 0 | 24.628 | 32 (0,01) | 1 | 201 | 4.899 | 9.788 | 10.322,6 | 15.313 | 23.692 | 24.628 |
| `mes` | int | int64 | 0 | 11 | 202601 (9,45) | 202601 | — | 202603 | 202606 | 202605,98 | 202609 | 202611 | 202611 |
| `edad` | int | int64 | 0 | 50 | 39 (2,23) | 22 | 22 | 34 | 46 | 46,36 | 59 | 71 | 71 |
| `ingresos` | float | float64 | 0 | 24.232 | 18000 (1,63) | 18.000 | 18.000 | 50.148,6 | 64.858,0 | 65.162,5 | 79.838,1 | 116.247,4 | 145.700,1 |
| `ratio_deuda_ingresos` | float | float64 | 0 | 24.582 | 0,02 (0,15) | 0,02 | 0,044 | 0,201 | 0,311 | 0,329 | 0,438 | 0,772 | 0,95 |
| `antiguedad_cuenta_meses` | int | int64 | 0 | 179 | 43 (0,73) | 1 | 2 | 44 | 88 | 88,90 | 133 | 178 | 179 |
| **`numero_productos`** | int | int64 | 0 | **5** | **1 (45,66)** | 1 | 1 | 1 | 2 | 1,923 | 3 | 5 | 5 |
| `saldo_promedio` | float | float64 | 0 | 22.985 | 300 (6,86) | 300 | 300 | 14.487,4 | 25.692,0 | 26.294,2 | 37.264,1 | 65.552,5 | 90.000 |
| **`dias_ultima_transaccion`** | int | int64 | 0 | 364 | 313 (0,41) | 1 | 4 | 101 | 194 | 190,25 | 282 | 361 | 364 |
| `ocupacion` | string | object | 0 | 5 | clerical (30,01) | | | | | | | | |
| `region` | string | object | 0 | 5 | west (25,06) | | | | | | | | |
| `canal_adquisicion` | string | object | 0 | 5 | web (33,06) | | | | | | | | |
| **`banda_riesgo`** | string | object | 0 | 3 | low (46,77) | | | | | | | | |
| **`dias_ultima_interaccion`** | int | int64 | 0 | 364 | 313 (0,42) | 1 | 4 | 100 | 194 | 190,16 | 282 | 361 | 364 |
| `tiene_tarjeta_credito` | bool | bool | 0 | 2 | True (74,19) | | | | | | | | |
| `activo_movil` | bool | bool | 0 | 2 | True (65,57) | | | | | | | | | |
| `es_nuevo_cliente` | bool | bool | 0 | 2 | False (80,79) | | | | | | | | | |
| `tiene_prestamo` | bool | bool | 0 | 2 | False (55,41) | | | | | | | | | |
| `antiguedad_direccion_meses` | int | int64 | 0 | 239 | 88 (0,63) | 1 | 3 | 61 | 120 | 119,96 | 179 | 237 | 239 |
| `visitas_web_ultimos_90_dias` | int | int64 | 0 | 23 | 8 (14,48) | 0 | 2 | 6 | 8 | 8,01 | 10 | 15 | 26 |
| `distancia_sucursal_km` | float | float64 | 0 | 24.588 | 80 (0,14) | 0,2125 | 1,178 | 4,331 | 7,487 | 10,158 | 12,718 | 47,939 | 80 |
| `dispositivo_principal` | string | object | 0 | 4 | android (42,00) | | | | | | | | |
| `dia_preferido_pago` | int | int64 | 0 | 28 | 25 (3,86) | 1 | 1 | 8 | 15 | 14,53 | 22 | 28 | 28 |
| `tiene_seguro` | bool | bool | 0 | 2 | False (61,72) | | | | | | | | | |
| `objetivo` | bool | int64 | 0 | 2 | 0 (84,95) | 0 | 0 | 0 | 0 | 0,15 | 0 | 1 | 1 |
